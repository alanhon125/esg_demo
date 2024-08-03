import io
import math
import PIL
import pandas as pd
from PIL import Image, ImageDraw
from grits import objects_to_cells

import fitz
fitz.TOOLS.set_small_glyph_heights(True)


def get_page_tokens(opened_pdf, page_no):
    page = opened_pdf[page_no]
    raw_words = page.get_text("words")
    
    tokens = [{
        "text": w[4],
        "bbox": w[:4],
        "flags": 0, 
        "block_num": w[5], 
        "line_num": w[6], 
        "span_num": w[7]
    } for w in raw_words]

    sc_tokens = get_page_superscript_tokens(opened_pdf, page_no)

    tokens_dict = {}
    for t in tokens:
        if (t["block_num"], t["line_num"]) in tokens_dict:
            tokens_dict[(t["block_num"], t["line_num"])].append(t)
        else:
            tokens_dict[(t["block_num"], t["line_num"])] = [t]

    for t in sc_tokens:
        if (t["block_num"], t["line_num"]) in tokens_dict:
            for tt in tokens_dict[(t["block_num"], t["line_num"])]:
                if fitz.Rect(t["bbox"]).intersects(tt["bbox"]):
                    sc_text = t["text"].strip()
                    start = tt["text"].find(sc_text)

                    # if start == -1:
                    #     print(tt)
                    #     print(t)
                    #     print("="*30)
                    end = start + len(sc_text)
                    if start > -1:
                        tt["text"] = f'{tt["text"][:start]}<s>{sc_text}</s>{tt["text"][end:]}'
                    
    return tokens


def get_page_superscript_tokens(opened_pdf, page_no):
    page = opened_pdf[page_no]
    page_dict = page.get_text("dict")

    tokens = []

    for i in range(len(page_dict["blocks"])):
        block = page_dict["blocks"][i]
        if "lines" in block.keys():
            for j in range(len(block["lines"])):
                line = block["lines"][j]
                if "spans" in line.keys():
                    for k in range(len(line["spans"])):
                        span = line["spans"][k]
                        if span["flags"] & 2**0:
                            sc_text_list = span["text"].strip().split(" ")
                            for sc in sc_text_list:
                                tokens.append({
                                    "text": sc,
                                    "bbox": span["bbox"],
                                    "flags": 0,
                                    "block_num": i, 
                                    "line_num": j, 
                                    "span_num": k
                                })

    return tokens


# resize the image
def resize_img(pil_img, max_width=1000, max_height=1000):
    width, height = pil_img.size
    ratio = min(max_width/width, max_height/height)
    if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
        PIL.Image.Resampling = PIL.Image
    result = pil_img.resize((math.floor(width*ratio), math.floor(height*ratio)))
    return result, ratio


def add_margin(pil_img, top, right, bottom, left, color=(255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def scale_bbox_to_origin(bbox, margin_w, margin_h, ratio, origin=None):
    scaled_bbox = bbox[:]
    scaled_bbox[0] = max(0, (scaled_bbox[0] - margin_w)/ratio)
    scaled_bbox[1] = max(0, (scaled_bbox[1] - margin_h)/ratio)
    scaled_bbox[2] = max(0, (scaled_bbox[2] - margin_w)/ratio)
    scaled_bbox[3] = max(0, (scaled_bbox[3] - margin_h)/ratio)
    
    if origin is not None:
        xmin, ymin = origin
        scaled_bbox[0] += xmin
        scaled_bbox[1] += ymin
        scaled_bbox[2] += xmin
        scaled_bbox[3] += ymin

    return scaled_bbox


def table_to_df(model, opened_pdf, table):
    try:
        _, cells, _ = table_to_cells(model, opened_pdf, table)
        df = cells_to_df(cells)
        return df
    except:
        return pd.DataFrame()


TABLE_RECOGNITION_MAX_WIDTH = 1000
TABLE_RECOGNITION_MAX_HEIGHT = 1000
TABLE_RECOGNITION_MARGIN_RATIO = 0.055



def table_to_cells(model, opened_pdf, table):
    page_no = table["page_no"]
    bbox = table["enlarged_bbox"]

    tokens = get_page_tokens(opened_pdf, page_no)
    tokens = [t for t in tokens 
                if fitz.Rect(t["bbox"]).intersects(bbox)]
    
    structure_class_names = [
        "table",
        "table column",
        "table row",
        "table column header",
        "table projected row header",
        "table spanning cell",
        "no object",
    ]
    structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
    structure_class_thresholds = {
            "table": 0.9,
            "table column": 0.8,
            "table row": 0.6,
            "table column header": 0.5,
            "table projected row header": 0.8,
            "table spanning cell": 0.8,
            "no object": 10,
        }
    
    byte = opened_pdf[page_no].get_pixmap().pil_tobytes(format="JPEG", optimize=True)
    f = io.BytesIO(byte)
    img = Image.open(f).convert("RGB")
    tab_img = img.crop(bbox)

    scaled_img, ratio = resize_img(tab_img, max_width=TABLE_RECOGNITION_MAX_WIDTH, max_height=TABLE_RECOGNITION_MAX_HEIGHT)
    w, h = scaled_img.size
    margin_w = int(w * TABLE_RECOGNITION_MARGIN_RATIO)
    margin_h = int(h * TABLE_RECOGNITION_MARGIN_RATIO)
    margin = max(margin_w, margin_h)
    scaled_img = add_margin(scaled_img, margin, margin, margin, margin, (255,255,255,255))
    res = model.predict(scaled_img, debug=False)

    scores = list(map(float, res["scores"]))
    labels = list(map(int, res["labels"]))
    pred_boxes = res["boxes"]
    processed_boxes = []

    origin = (bbox[0], bbox[1])

    for i in range(len(pred_boxes)):
        temp_box = list(map(float, pred_boxes[i]))
        processed_boxes.append(scale_bbox_to_origin(temp_box, margin, margin, ratio, origin))
        
    return objects_to_cells(processed_boxes, labels, scores, tokens, structure_class_names, structure_class_thresholds, structure_class_map)


def cells_to_df(cells):
    data_dict = parse_all(cells)
    return pd.DataFrame(data=data_dict)
    

def get_header_cells(cells):
    return [c for c in cells if c["header"] == True]


def get_subheader_cells(cells):
    return [c for c in cells if c["subheader"] == True]


def get_cells_by_row(cells, row_no):
    return [c for c in cells if row_no in c["row_nums"]]


def get_cells_by_col(cells, col_no):
    return [c for c in cells if col_no in c["column_nums"]]


def get_row_col_num(cells):
    cols, rows = set(), set()
    for c in cells:
        for cid in c["column_nums"]:
            cols.add(cid)
        for rid in c["row_nums"]:
            rows.add(rid)
    return len(rows), len(cols)


def is_empty_row(row_cells):
    empty_flag = True    
    for cell in row_cells:
        if len(cell["cell_text"]) > 0:
            empty_flag = False
    return empty_flag


def parse_header(cells):
    header_cells = get_header_cells(cells)
    row_num, col_num = get_row_col_num(header_cells)

    parsed_header = []

    for cid in range(col_num):
        cells_cid = get_cells_by_col(header_cells, cid)
        cells_cid.sort(key=lambda x: x["bbox"][1])
        header_text = []
        for c in cells_cid:
            header_text.append(c["cell_text"])
        parsed_header.append(" ".join(header_text))

    return parsed_header, row_num, col_num


def find_sub_header(row_cells):
    for rc in row_cells:
        if rc["subheader"]:
            return rc
    return False


def is_probably_row_header(row_cells, length_thresh=3):
    if len(row_cells) > length_thresh and len(row_cells[0]["cell_text"]) > 0:
        for c in row_cells[1:]:
            if len(c["cell_text"]) > 0:
                return False
        return True
    return False


def get_cell_text(row_cells, cid):
    for cell in row_cells:
        if cid in cell["column_nums"]:
            return cell["cell_text"]
    raise Exception(f"cid {cid} not found in row cells")


def parse_all(cells):
    parsed_header, col_row_num, col_col_num = parse_header(cells)
    row_num, col_num = get_row_col_num(cells)

    if parsed_header == [] and col_col_num == 0 and col_row_num == 0:
        parsed_header = []
        for i in range(col_num):
            parsed_header.append(f"col{i}")
        col_col_num = len(parsed_header)
        col_row_num = 0

    # assert col_num == col_col_num

    data = [[] for _ in parsed_header]
    
    prefix = ""

    for rid in range(col_row_num, row_num):
        row_cells = get_cells_by_row(cells, rid)
        row_cells.sort(key=lambda rc: rc["column_nums"][0])
        if find_sub_header(row_cells):
            prefix = find_sub_header(row_cells)["cell_text"]
        elif is_probably_row_header(row_cells):
            prefix = row_cells[0]["cell_text"]
        else:
            if is_empty_row(row_cells):
                continue
            for cid in range(col_num):
                try:
                    if cid == 0 and len(prefix) > 0:
                        data[cid].append(f"*{prefix}*{get_cell_text(row_cells, cid)}")
                    elif cid == 0 and len(row_cells[cid]['row_nums'])>1:
                        data[cid].append(f'{get_cell_text(row_cells, cid)}_{0}')
                    else:
                        data[cid].append(get_cell_text(row_cells, cid))
                except:
                    print((rid, cid))
                    
    
    data_dict = {}
    
    for i in range(len(parsed_header)):
        key = parsed_header[i]
        while key in data_dict:
            key += "*"
        data_dict[key] = data[i]

        # if parsed_header[i] not in data_dict:
        #     data_dict[parsed_header[i]] = data[i]
        # else:
        #     data_dict[parsed_header[i] + "*"] = data[i]
    
    assert len(data_dict.keys()) == col_num
        
    return data_dict


def vis_cells(img, cells, copy=True):
    if copy:
        vis_img = img.copy()
    else:
        vis_img = img
    draw = ImageDraw.Draw(vis_img)

    row_num, _ = get_row_col_num(cells)
    refined_cells = []
    for rid in range(row_num):
        row_cells = get_cells_by_row(cells, rid)
        if not is_empty_row(row_cells):
            refined_cells.extend(row_cells)
    
    for c in refined_cells:
        if c["header"]:
            draw.rectangle(c["bbox"], outline="blue")
        elif c["subheader"]:
            draw.rectangle(c["bbox"], outline="green")
        else:
            draw.rectangle(c["bbox"], outline="red")
        # text_pos = (c["bbox"][2], c["bbox"][1])

    return vis_img
            

def try_find_table_caption(opened_pdf, tab_info, search_range=40):
    page_no = tab_info["page_no"]
    
    bbox = tab_info["enlarged_bbox"].copy()
    bbox[3] = bbox[1]
    bbox[1] = max(0, bbox[1] - search_range)    
    
    tokens = get_page_tokens(opened_pdf, page_no)
    raw_caption = [t["text"] for t in tokens if fitz.Rect(t["bbox"]).intersects(bbox)]

    return "".join(raw_caption)



