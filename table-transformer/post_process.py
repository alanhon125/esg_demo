import json
import sys
import io
import fitz

from PIL import Image, ImageDraw


def post_process(path, result, left_bottom=True):
    doc = fitz.open(path + result["filename"])
    _, _, w, h = doc[0].rect
    bbox = result["table_areas"]
    scale_ratio = result["scale_ratio"]
    bbox = [x/scale_ratio for x in bbox]
    new_bbox = [bbox[0], h-bbox[1], bbox[2], h-bbox[3]]
    if left_bottom:
        return {
            "filename": result["filename"],
            "page_no": result["page_no"],
            "table_areas": new_bbox
        }
    else:
        return {
            "filename": result["filename"],
            "page_no": result["page_no"],
            "table_areas": bbox
        }


def visualize_result(path, result, i=None):
    doc = fitz.open(path + result["filename"])
    page = doc[result["page_no"]]
    byte = page.get_pixmap().pil_tobytes(format="JPEG", optimize=True)
    f = io.BytesIO(byte)
    img = Image.open(f)
    draw = ImageDraw.Draw(img)
    draw.rectangle(result["table_areas"], outline ="red")
    img.save(f"{result['filename']}_{result['page_no']}_table_{i}.jpeg")


if __name__== "__main__":
    path = "/home/data/lqy/2021/"
    
    with open(path + "result.json", "r") as f:
        results = json.load(f)

    for i in range(15):
        visualize_result(path, results[i], i)
