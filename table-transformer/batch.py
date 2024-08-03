from core import TableDetector
from utils import pdf2images, resize_img
from PIL import Image
import os

import json


def process_result(results, new_results, fname, page_no, ratio, thresh=0.9, height=None):
    for idx, score in enumerate(results["scores"].tolist()):
        if score < thresh:
            continue
        xmin, ymin, xmax, ymax = list(map(float, results["boxes"][idx]))
        temp = {
            "filename": fname,
            "page_no": page_no,
            "table_areas": [xmin/ratio, ymin/ratio, xmax/ratio, ymax/ratio]
        }
        if height is not None:
            temp["table_areas"][1] = height - temp["table_areas"][1]
            temp["table_areas"][3] = height - temp["table_areas"][3]

        new_results.append(temp)


import sys

def run(path, output, thresh, model):
    processed = []
    all_files = os.listdir(path)
    cnt = 1
    for fname in all_files:
        print(f"{cnt}/{len(all_files)} " + path + fname, end="\r", flush=True)
        cnt = cnt + 1
        img_list = pdf2images(path + fname)
        for idx in range(len(img_list)):
            img = img_list[idx]
            original_height = img.height
            img_new, ratio = resize_img(img, 800, 800)
            results = model.predict(img_new, thresh=thresh, debug=False)
            process_result(results, processed, fname, idx, ratio, thresh=thresh, height=original_height)
            # process_result(results, processed, fname, idx, ratio, thresh=thresh)
    with open(output, 'w') as json_file:
        json.dump(processed, json_file, indent=4, separators=(',',': '))


if __name__ == "__main__":
    m = TableDetector(checkpoint_path="/home/liuqy/table_extraction/table-transformer/pubtables1m_detection_detr_r18.pth")

    path = sys.argv[1]
    output = sys.argv[2]
    thresh = float(sys.argv[3])

    run(path, output, thresh, m)
    
