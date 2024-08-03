import utils
from PIL import Image, ImageDraw


def draw_bbox(img: Image, bbox: list, color: str="red"):
    assert len(bbox) == 4
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=color)


def draw_text(img: Image, pos: list, text: str, color: str="red"):
    assert len(pos) == 2
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, fill=color)


def draw_bboxes(img: Image, bboxes: list, color: str="red"):
    for bbox in bboxes:
        draw_bbox(img, bbox, color)


def vis_all_tables(pdf_img: Image, page_no: int, tables: list, bbox_type: str="enlarged_bbox"):
    # do not change the original pdf image
    img = pdf_img.copy()

    found_table = False
    for tab in tables:
        if tab["page_no"] == page_no:
            found_table = True
            bbox = tab[bbox_type]
            draw_bbox(img, bbox, color="red")
            pos = [bbox[2], bbox[1]]
            draw_text(img, pos, f"{tab['score']:.3f}")

    if found_table:
        return img
    else:
        return None




