import fitz
import math
import io
import copy
import pandas as pd
import PIL
from PIL import Image, ImageDraw
from scipy import spatial
from visualization import vis_all_tables


# utilities to handle pdf operations
# extract table from a openned pdf file by providing the page no and a bbox
def extract_table_from_pdf(doc, page_no, rect):
    # doc = fitz.open(fname)
    page = doc[page_no]
    pix = page.get_pixmap(clip=rect)
    return pix


def pdf2images(fname, if_bytes=False, keywords=["emission", "ghg", "scope"]):
    if if_bytes:
        doc = fitz.open(stream=fname)
    else:
        doc = fitz.open(fname)
    page_cnt = doc.page_count
    img_list = []
    page_no_list = []
    for i in range(page_cnt):
        flag = False
        plain_text = doc[i].get_text().lower()
        for keyword in keywords:
            if plain_text.find(keyword) > -1:
                flag = True
                continue
        if flag:
            byte = doc[i].get_pixmap().pil_tobytes(format="JPEG", optimize=True)
            f = io.BytesIO(byte)
            img = Image.open(f).convert("RGB")
            img_list.append(img)
            page_no_list.append(i)
    return img_list, page_no_list


def pdf2images_all(fname, if_bytes=False):
    if if_bytes:
        doc = fitz.open(stream=fname)
    else:
        doc = fitz.open(fname)
    page_cnt = doc.page_count
    img_list = []
    page_no_list = []
    for i in range(page_cnt):
        byte = doc[i].get_pixmap().pil_tobytes(format="JPEG", optimize=True)
        f = io.BytesIO(byte)
        img = Image.open(f).convert("RGB")
        img_list.append(img)
        page_no_list.append(i)
    return img_list, page_no_list


def check_bbox(bbox, w, h):
    xmin, ymin, xmax, ymax = bbox
    return xmin > 0 and ymin > 0 and xmax < w and ymax < h and xmin < xmax and ymin < ymax


# margin added to the detected bbox area
TABLE_DETECTION_MARGIN = 5

def table_detection(fname, model, threshold, 
    scale=[800, 800],
    keywords=None, 
    anchor_words=[],
    caption_words=[],
    make_symmetric=True,
    debug=False,
    debug_output_path=None
    ):

    if keywords is None:
        img_list, page_no_list = pdf2images_all(fname)
    elif len(keywords) == 0:
        img_list, page_no_list = pdf2images_all(fname)
    else:
        img_list, page_no_list = pdf2images(fname, keywords=keywords)

    tables = []
    doc = fitz.open(fname)
    for i in range(len(img_list)):
        w, h = img_list[i].size
        img = img_list[i]
        img, ratio = resize_img(img, scale[0], scale[1])
        result = model.predict(img, thresh=threshold)
        temp_tables = []
        for idx, score in enumerate(result["scores"].tolist()):
            if score < threshold:
                continue
            bbox = list(map(float, result["boxes"][idx]))
            bbox_add_margin = [x/ratio for x in bbox]
            # bbox_add_margin = bbox
            bbox_add_margin[0] = max(0, bbox_add_margin[0] - TABLE_DETECTION_MARGIN)
            bbox_add_margin[1] = max(0, bbox_add_margin[1] - TABLE_DETECTION_MARGIN)
            bbox_add_margin[2] = min(w, bbox_add_margin[2] + TABLE_DETECTION_MARGIN)
            bbox_add_margin[3] = min(h, bbox_add_margin[3] + TABLE_DETECTION_MARGIN)
            tab = {"fname": fname,
                   "page_no": page_no_list[i], 
                   "score": score,
                   "bbox": bbox_add_margin,
                   "raw_bbox": bbox,
                  }
            enlarge_table(doc, tab, anchor_words=anchor_words, caption_words=caption_words, make_symmetric=make_symmetric)
            xmin, ymin, xmax, ymax = tab["enlarged_bbox"]
            tab["table_areas"] = ",".join([str(xx) for xx in [xmin, h-ymin, xmax, h-ymax]])
            if check_bbox(tab["enlarged_bbox"], w, h):
                temp_tables.append(tab)
        
        if len(result) > 0:
            pad_img = pad_img_with_bboxes(img, 
                [b["enlarged_bbox"] for b in temp_tables])
            pad_result = model.predict(pad_img, thresh=threshold)
            for idx, score in enumerate(pad_result["scores"].tolist()):
                if score < threshold:
                    continue
                bbox = list(map(float, pad_result["boxes"][idx]))
                bbox_add_margin = [x/ratio for x in bbox]
                # bbox_add_margin = bbox
                bbox_add_margin[0] = max(0, bbox_add_margin[0] - TABLE_DETECTION_MARGIN)
                bbox_add_margin[1] = max(0, bbox_add_margin[1] - TABLE_DETECTION_MARGIN)
                bbox_add_margin[2] = min(w, bbox_add_margin[2] + TABLE_DETECTION_MARGIN)
                bbox_add_margin[3] = min(h, bbox_add_margin[3] + TABLE_DETECTION_MARGIN)
                tab = {"fname": fname,
                       "page_no": page_no_list[i], 
                       "score": score,
                       "bbox": bbox_add_margin,
                       "raw_bbox": bbox,
                      }
                enlarge_table(doc, tab, anchor_words=anchor_words, caption_words=caption_words, make_symmetric=make_symmetric)
                xmin, ymin, xmax, ymax = tab["enlarged_bbox"]
                tab["table_areas"] = ",".join([str(xx) for xx in [xmin, h-ymin, xmax, h-ymax]])
                temp_tables.append(tab)
        
        remove_list = []
        if len(temp_tables) > 1:
            for i in range(len(temp_tables)):
                box1 = temp_tables[i]["enlarged_bbox"]
                for j in range(i+1, len(temp_tables)):
                    box2 = temp_tables[j]["enlarged_bbox"]
                    area1 = get_area(box1)
                    area2 = get_area(box2)
                    if area1 > 0 and area2 > 0 and get_intersection_area(box1, box2) / area1 > 0.3:
                        if temp_tables[i]["score"] > temp_tables[j]["score"]:
                            remove_list.append(j)
                        else:
                            remove_list.append(i)

        for i in range(len(temp_tables)):
            if i in remove_list:
                continue
            else:
                tables.append(temp_tables[i])

    if debug:
        if debug_output_path is None:
            debug_output_path = "./debug"
        
        if not os.path.exists(debug_output_path):
            os.mkdir(debug_output_path)
        else:
            os.system(f"rm {debug_output_path}/*")
        for idx in range(len(page_no_list)):
            page_no = page_no_list[idx]
            pdf_img = img_list[idx]
            draw = vis_all_tables(pdf_img, page_no, tables)
            if draw is not None:
                draw.save(f"{debug_output_path}/{page_no}.png")

    return tables, img_list, page_no_list


ENLARGE_TABLE_MARGIN = 4

# this function is to make sure the completeness of a detected table
# the anchor words are assumed to be in the first row (i.e., column header)
def enlarge_table(doc, table, 
    all_words=None, 
    anchor_words=[], 
    caption_words=[],
    make_symmetric=False):

    page_no = table["page_no"]
    page = doc[page_no]
    
    if all_words is None:
        all_words = extract_text_from_pdf(doc, page_no)
    
    words_in_bbox = [w for w in all_words if fitz.Rect(w[:4]).intersects(table["bbox"])]
    
    if len(words_in_bbox) == 0:
        table["enlarged_bbox"] = table["bbox"]
        return
    
    xmin, ymin, xmax, ymax = 99999, 99999, -1, -1
    
    for word in words_in_bbox:
        xmin = min(xmin, word[0])
        ymin = min(ymin, word[1])
        xmax = max(xmax, word[2])
        ymax = max(ymax, word[3])

    # 
    if len(caption_words) > 0 and table["score"]<0.9:
        for a in caption_words:
            for w in words_in_bbox:
                if w[4].find(a) != -1:
                    ymin = w[3]

    new_bbox = [xmin, ymin, xmax, ymax]
    
    if len(anchor_words) > 0:
        flag = False
        for a in anchor_words:
            for w in words_in_bbox:
                if w[4].find(a) != -1:
                    flag = True
        # only enlarge space if the table does not cover anchor words
        if flag == False:
            h = (ymax - ymin) / 3.0
            est_anchor_words_range = [xmin, max(0, ymin-h), xmax, ymin]
            candidate_anchor_words = [w for w in all_words if 
                fitz.Rect(w[:4]).intersects(est_anchor_words_range)]
            anchor_word_locs = []
            for a in anchor_words:
                temp_candidates = []
                for candidate in candidate_anchor_words:
                    if candidate[4].find(a) != -1:
                        temp_candidates.append(candidate)
                if len(temp_candidates) > 0:
                    temp_candidates.sort(key=lambda x: x[1])
                    anchor_word_locs.append(temp_candidates[0])
            if len(anchor_word_locs) > 0:
                for a in anchor_word_locs:
                    new_bbox[1] = min(new_bbox[1], a[1])

    if make_symmetric:
        w, h = page.mediabox_size
        left_margin = new_bbox[0]
        right_margin = w - new_bbox[2]
        if left_margin < right_margin:
            new_bbox[2] = w - left_margin
        
    new_bbox[0] -= ENLARGE_TABLE_MARGIN
    new_bbox[1] -= ENLARGE_TABLE_MARGIN
    new_bbox[2] += ENLARGE_TABLE_MARGIN
    new_bbox[3] += ENLARGE_TABLE_MARGIN

    table["enlarged_bbox"] = new_bbox


def pad_img_with_bboxes(img_to_pad, bboxes, color=(255,255,255)):
    img = img_to_pad.copy()
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, fill=color)
    return img


def extract_table_img(img_list, page_no, bbox):
    img = img_list[page_no].copy()
    return img.crop(bbox)


# extract *ALL* words from a openned pdf file by providing the page no
def extract_text_from_pdf(doc, page_no):
    page = doc[page_no]
    words = page.get_text("words")
    return words


# filter words by bbox
def get_words_by_rect(words, rect, allow_intersection=True, match_area=True, split=" "):
    if allow_intersection:
        if match_area:
            words_in_rect = [w for w in words 
                if get_intersection_area(rect, fitz.Rect(w[:4]))/get_area(fitz.Rect(w[:4]))>0.7]
        else:
            words_in_rect =  [w for w in words if fitz.Rect(w[:4]).intersects(rect)]
    else:
        words_in_rect =  [w for w in words if fitz.Rect(w[:4]) in rect]
    combined = [w[4] for w in words_in_rect]
    return split.join(combined)


def get_area(box):
    dx = box[2] - box[0]
    dy = box[3] - box[1]
    if dx>0 and dy>0:
        return dx * dy
    else:
        return 0


def get_intersection_area(box1, box2):
    dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
    dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if dx>0 and dy>0:
        return dx * dy
    else:
        return 0


# utilities to pre-process input data
# add blank margin to the original input image
def add_margin(pil_img, top, right, bottom, left, color=(255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


# resize the image
def resize_img(pil_img, max_width=1000, max_height=1000):
    width, height = pil_img.size
    ratio = min(max_width/width, max_height/height)
    if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
        PIL.Image.Resampling = PIL.Image
    result = pil_img.resize((math.floor(width*ratio), math.floor(height*ratio)))
    return result, ratio


# add margin & resize
def preprocess_img(pil_img, margin=50, max_width=1000, max_height=1000):
    result, ratio = resize_img(pil_img, max_width, max_height)
    result = add_margin(result, margin, margin, margin, margin)
    return result, ratio


def scale_bbox_to_origin(bbox, margin, ratio, origin=None):
    scaled_bbox = [max(0, (x - margin)/ratio) for x in bbox]
    
    if origin is not None:
        xmin, ymin = origin
        scaled_bbox[0] += xmin
        scaled_bbox[1] += ymin
        scaled_bbox[2] += xmin
        scaled_bbox[3] += ymin
    return scaled_bbox


# utilities to post-process the model output
def find_most_similar_bbox(query_bbox, bbox_list):
    cos_dist_list = [spatial.distance.cosine(query_bbox, bbox) for bbox in bbox_list]
    idx = min(range(len(cos_dist_list)), key=cos_dist_list.__getitem__)
    return idx


def align_table_elements(element_dict: dict):
    # sort rows by y-axis
    element_dict["row"].sort(key=lambda x: x[1])
    element_dict["projected_row_header"].sort(key=lambda x: x[1])

    # sort columns by x-axis
    element_dict["column"].sort(key=lambda x: x[0])
    element_dict["column_header"].sort(key=lambda x: x[0])

    # sort spanning cells by x-axis
    element_dict["spanning_cell"].sort(key=lambda x: x[0])

    # find xmin, ymin, xmax, ymax
    xmin_1 = min([x[0] for x in element_dict["row"]])
    xmin_2 = min([x[0] for x in element_dict["column"]])
    xmin = min(element_dict["table"][0][0], xmin_1, xmin_2)

    ymin_1 = min(x[1] for x in element_dict["row"])
    ymin_2 = min(x[1] for x in element_dict["column"])
    ymin = min(element_dict["table"][0][1], ymin_1, ymin_2)

    xmax_1 = max([x[2] for x in element_dict["row"]])
    xmax_2 = max([x[2] for x in element_dict["column"]])
    xmax = max(element_dict["table"][0][2], xmax_1, xmax_2)

    ymax_1 = max(x[3] for x in element_dict["row"])
    ymax_2 = max(x[3] for x in element_dict["column"])
    ymax = max(element_dict["table"][0][3], ymax_1, ymax_2)

    # align all rows
    for row in element_dict["row"]:
        row[0] = xmin
        row[2] = xmax
    
    # align first row
    element_dict["row"][0][1] = ymin
    
    # align last row
    element_dict["row"][-1][3] = ymax

    # align all projected_row_header
    for row in element_dict["projected_row_header"]:
        row[0] = xmin
        row[2] = xmax

    # align all columns
    for col in element_dict["column"]:
        col[1] = ymin
        col[3] = ymax
    
    # align first column
    element_dict["column"][0][0] = xmin

    # align last column
    element_dict["column"][-1][2] = xmax

    # align column_header
    if len(element_dict["column_header"]) == 1:
        element_dict["column_header"][0][0] = xmin
        element_dict["column_header"][0][1] = ymin
        element_dict["column_header"][0][2] = xmax
    elif len(element_dict["column_header"]) == 0:
        element_dict["column_header"] = [copy.deepcopy(element_dict["row"][0])]
    else:
        # should not happen
        print(len(element_dict["column_header"]))
        raise RuntimeError("strange column header!")

    # align table box
    element_dict["table"][0][0] = xmin
    element_dict["table"][0][1] = ymin
    element_dict["table"][0][2] = xmax
    element_dict["table"][0][3] = ymax

    # resolve intersected rows
    for i in range(len(element_dict["row"]) - 1):
        y1_max = element_dict["row"][i][3]
        y2_min = element_dict["row"][i+1][1]
        y_mid = (y1_max + y2_min) / 2.0
        element_dict["row"][i][3] = y_mid
        element_dict["row"][i+1][1] = y_mid

    # resolve intersected columns
    for i in range(len(element_dict["column"]) - 1):
        x1_max = element_dict["column"][i][2]
        x2_min = element_dict["column"][i+1][0]
        x_mid = (x1_max + x2_min) / 2.0
        element_dict["column"][i][2] = x_mid
        element_dict["column"][i+1][0] = x_mid

    # match projected_row_header to row based on cosine similarity
    for i in range(len(element_dict["projected_row_header"])):
        idx = find_most_similar_bbox(element_dict["projected_row_header"][i], element_dict["row"])
        element_dict["projected_row_header"][i] = copy.deepcopy(element_dict["row"][idx])

    # TODO: align spanning cells



def postprocess_output(res, margin, ratio, thresh=0.9, align=True, origin=None):
    temp = {
        "table": [],
        "column": [],
        "row": [],
        "column_header": [],
        "projected_row_header": [],
        "spanning_cell": []
    }
    for idx, score in enumerate(res["scores"].tolist()):
        if score < thresh:
            continue
        bbox = list(map(float, res["boxes"][idx]))
        label = res["labels"][idx].item()
        if (label == 0):
            temp["table"].append(scale_bbox_to_origin(bbox, margin, ratio))
        elif (label == 1):
            temp["column"].append(scale_bbox_to_origin(bbox, margin, ratio))
        elif (label == 2):
            temp["row"].append(scale_bbox_to_origin(bbox, margin, ratio))
        elif (label == 3):
            temp["column_header"].append(scale_bbox_to_origin(bbox, margin, ratio))
        elif (label == 4):
            temp["projected_row_header"].append(scale_bbox_to_origin(bbox, margin, ratio))
        elif (label == 5):
            temp["spanning_cell"].append(scale_bbox_to_origin(bbox, margin, ratio))

    if align:
        align_table_elements(temp)
    
    if origin is not None:
        o_xmin, o_ymin = origin
        for key in temp.keys():
            for i in range(len(temp[key])):
                temp[key][i][0] = temp[key][i][0] + o_xmin
                temp[key][i][1] = temp[key][i][1] + o_ymin
                temp[key][i][2] = temp[key][i][2] + o_xmin
                temp[key][i][3] = temp[key][i][3] + o_ymin
            
    return temp


TABLE_RECOGNITION_MARGIN = 55

TABLE_RECOGNITION_MAX_WIDTH = 1000
TABLE_RECOGNITION_MAX_HEIGHT = 1000

def table_recogonition(model, img, threshold=0.9, align=True, origin=None):
    scaled_img, ratio = resize_img(img, max_width=TABLE_RECOGNITION_MAX_WIDTH, max_height=TABLE_RECOGNITION_MAX_HEIGHT)
    scaled_img = add_margin(scaled_img, TABLE_RECOGNITION_MARGIN, TABLE_RECOGNITION_MARGIN, TABLE_RECOGNITION_MARGIN, TABLE_RECOGNITION_MARGIN, (255,255,255,255))
    result = model.predict(scaled_img, thresh=threshold)
    return postprocess_output(result, TABLE_RECOGNITION_MARGIN, ratio, threshold, align, origin)


def get_intersection(col, row):
    return [col[0], row[1], col[2], row[3]]


def intersects_by_area(box1, box2, thresh=0.5):
    f_box1 = fitz.Rect(box1)

    if f_box1.intersects(box2):
        intersect = copy.deepcopy(f_box1).intersect(box2)
        area_box1 = f_box1.height * f_box1.width
        area_intersect = intersect.height * intersect.width
        return area_intersect/area_box1 > thresh
    else:
        return False


def parse_column_header(table, recogonition_res, all_words):
    if len(recogonition_res["column_header"]) == 1:
        col_header_row = recogonition_res["column_header"][0]
        rows = recogonition_res["row"]
        cols = recogonition_res["column"]

        col_header_rows = []
        for row in rows:
            col_header_rows.append(row)
            if abs(row[3] - col_header_row[3]) < (rows[0][3] - rows[0][1]) / 3:
                break
        
        sp_cells = recogonition_res["spanning_cell"]
        col_sp_cells = [c for c in sp_cells if fitz.Rect(col_header_row).intersects(c)]
        col_sp_cells.sort(key=lambda x:x[0])

        if len(col_header_rows) == 1:
            first_level = []
            for col in cols:
                intersect_bbox = get_intersection(col, col_header_row)
                intersect_words = get_words_by_rect(all_words, intersect_bbox)
                first_level.append(intersect_words)
            return first_level, col_header_rows

        if len(col_header_rows) > 2:
            
            while True:
                if len(col_header_rows) == 2:
                    break
                col_header_rows.pop(0)
                rows.pop(0)

            if table["page_no"] == 6:
                print(col_header_rows)
                print(rows)

        assert(len(col_header_rows) == 2)

        if len(col_header_rows) == 2:
            first_level = []
            second_level = []
            cross_level = []
            for col in cols:
                intersect_bbox_1 = get_intersection(col, col_header_rows[0])
                intersect_words_1 = get_words_by_rect(all_words, intersect_bbox_1)
                first_level.append([intersect_bbox_1, intersect_words_1])

                intersect_bbox_2 = get_intersection(col, col_header_rows[1])
                intersect_words_2 = get_words_by_rect(all_words, intersect_bbox_2)
                second_level.append([intersect_bbox_2, intersect_words_2])

                intersect_bbox_3 = get_intersection(col, col_header_row)
                intersect_words_3 = get_words_by_rect(all_words, intersect_bbox_3)
                cross_level.append([intersect_bbox_3, intersect_words_3])
            
            assert len(first_level) == len(second_level)
            assert len(first_level) == len(cross_level)
  
            for i in range(len(col_sp_cells) - 1):
                if intersects_by_area(col_sp_cells[i], col_sp_cells[i+1], 0.3):
                    x_mid = max(col_sp_cells[i][2], col_sp_cells[i+1][0])
                    col_sp_cells[i][2] = x_mid
                    col_sp_cells[i+1][0] = x_mid

            word_bbox_in_sp_cells = []
            for cell in col_sp_cells:
                words_in_cell = get_words_by_rect(all_words, cell)
                word_bbox_in_sp_cells.append([cell, words_in_cell])
            
            # fill empties in the first level
            for i in range(len(first_level)):
                for sp in word_bbox_in_sp_cells:
                        if intersects_by_area(first_level[i][0], sp[0], 0.5):
                            first_level[i][1] = sp[1]
                            break
                # if first_level[i][1] == "":
                #     for sp in word_bbox_in_sp_cells:
                #         if intersects_by_area(first_level[i][0], sp[0], 0.5):
                #             first_level[i][1] = sp[1]
                #             break
                            

            joined_column_header = []
            for i in range(len(first_level)):
                if first_level[i][1] == second_level[i][1]:
                    joined_column_header.append(first_level[i][1])
                else:
                    joined_column_header.append(first_level[i][1] + "|" + second_level[i][1])

            assert len(joined_column_header) == len(cols)
            return joined_column_header, col_header_rows
            
    else:
        print(f"WARNING: no column header found! Page no {table['page_no']}")


def is_prob_proj_header(parsed_row):
    empty_cnt = 0
    for r in parsed_row:
        if r == "":
            empty_cnt = empty_cnt + 1
    # Rule 1: empty cells % > 70%
    if empty_cnt / (len(parsed_row) * 1.0) > 0.7:
        return True

    # Rule 2: 
    continous_empty_cnt = 1
    flag = False
    for r in parsed_row:
        if r == "":
            if flag:
                continous_empty_cnt = continous_empty_cnt + 1
            flag = True
    if flag == False:
        continous_empty_cnt = 0
    if continous_empty_cnt / (len(parsed_row) * 1.0) > 0.5:
        return True
    return False


def analyze_table_structure(pdf, table, recogonition_res):
    all_words = extract_text_from_pdf(pdf, table["page_no"])
    col_headers, col_header_rows = parse_column_header(table, recogonition_res, all_words)
    rows = recogonition_res["row"]
    cols = recogonition_res["column"]
    projected_row_headers = recogonition_res["projected_row_header"]

    parsed_table = []
    words_in_pj = ""
    for i in range(len(col_header_rows), len(rows)):
        row = rows[i]
        if row in projected_row_headers:
            words_in_pj = get_words_by_rect(all_words, row)
            continue
        parsed_row = []
        for j in range(len(cols)):
            cell = get_intersection(cols[j], rows[i])
            val = get_words_by_rect(all_words, cell)
            if j == 0 and words_in_pj != "":
                parsed_row.append(f"*{words_in_pj}*{val}")
            else:
                parsed_row.append(val)
        if is_prob_proj_header(parsed_row):
            projected_row_headers.append(row)
            words_in_pj = get_words_by_rect(all_words, row)
            continue
        parsed_table.append(parsed_row)

    # recogonition_res["projected_row_header"].sort(key=lambda x:x[1])
    return pd.DataFrame(data=parsed_table, columns=col_headers)


def simple_analze_table_structure(pdf, table, recogonition_res):
    all_words = extract_text_from_pdf(pdf, table["page_no"])
    rows = recogonition_res["row"]
    cols = recogonition_res["column"]

    parsed_table = []
    for i in range(len(rows)):
        parsed_row = []
        for j in range(len(cols)):
            cell = get_intersection(cols[j], rows[i])
            val = get_words_by_rect(all_words, cell, split="")
            parsed_row.append(val)
        parsed_table.append(parsed_row)

    return pd.DataFrame(data=parsed_table[1:], columns=parsed_table[0])


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
    return tokens


from grits import objects_to_cells
def table_to_cells(model, opened_pdf, table):
    page_no = table["page_no"]
    bbox = table["enlarged_bbox"]

    tokens = get_page_tokens(opened_pdf, page_no)
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
            "table": 0.5,
            "table column": 0.5,
            "table row": 0.5,
            "table column header": 0.5,
            "table projected row header": 0.5,
            "table spanning cell": 0.5,
            "no object": 10,
        }
    
    byte = opened_pdf[page_no].get_pixmap().pil_tobytes(format="JPEG", optimize=True)
    f = io.BytesIO(byte)
    img = Image.open(f).convert("RGB")
    tab_img = img.crop(bbox)

    scaled_img, ratio = resize_img(tab_img, max_width=TABLE_RECOGNITION_MAX_WIDTH, max_height=TABLE_RECOGNITION_MAX_HEIGHT)
    scaled_img = add_margin(scaled_img, TABLE_RECOGNITION_MARGIN, TABLE_RECOGNITION_MARGIN, TABLE_RECOGNITION_MARGIN, TABLE_RECOGNITION_MARGIN, (255,255,255,255))
    res = model.predict(scaled_img, debug=False)

    scores = list(map(float, res["scores"]))
    labels = list(map(int, res["labels"]))
    pred_boxes = res["boxes"]
    processed_boxes = []

    origin = (bbox[0], bbox[1])

    for i in range(len(pred_boxes)):
        temp_box = list(map(float, pred_boxes[i]))
        processed_boxes.append(scale_bbox_to_origin(temp_box, TABLE_RECOGNITION_MARGIN, ratio, origin))
        
    return objects_to_cells(processed_boxes, labels, scores, tokens, structure_class_names, structure_class_thresholds, structure_class_map)


def postprocess_cells():
    None


import traceback
def analyze_tables(model, fname, tables, thresh, simple=True):
    pdf = fitz.open(fname)
    img_list, _ = pdf2images_all(fname)

    all_res = []
    for tab in tables:
        bbox = tab["enlarged_bbox"]
        img = extract_table_img(img_list, tab["page_no"], bbox)
        try:
            recog_res = table_recogonition(model, img, threshold=thresh, origin=(bbox[0], bbox[1]), align=True)
            if simple:
                df = simple_analze_table_structure(pdf, tab, recog_res)
            else:
                df = analyze_table_structure(pdf, tab, recog_res)
            all_res.append((tab, df))
        except:
            # print(f"WARN: TSR failed on page no {tab['page_no']} score {tab['score']}")
            # traceback.print_exc()
            None
    
    return all_res


import pickle


def serialize(all_res, fname):
    with open(fname, "wb") as out:
        pickle.dump(all_res, out, pickle.HIGHEST_PROTOCOL)


import os
import shutil

def save_tsr_to_csv(all_res, path, folder_name):
    if not os.path.exists(path + folder_name):
        os.mkdir(path + folder_name)
    else:
        shutil.rmtree(path + folder_name)
        os.mkdir(path + folder_name)
    
    tab_cnt = 0
    last_page = -1
    for res in all_res:
        page_no = res[0]["page_no"]
        df = res[1]
        csv_name = str(page_no)
        if page_no == last_page:
            tab_cnt += 1
        else:
            tab_cnt = 0
            last_page = page_no
        csv_name += f"_{tab_cnt}.csv"
        df.to_csv(path + folder_name + "/" + csv_name, index=False)
