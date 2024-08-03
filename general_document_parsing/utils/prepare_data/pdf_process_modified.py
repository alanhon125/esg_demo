import multiprocessing
import argparse
import pdfplumber
import os
from tqdm import tqdm
from pdfminer.layout import LTChar, LTLine
import re
from collections import Counter
import pdf2image
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import nltk
import ndjson
import copy

def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95


def worker(pdf_file, data_dir, output_dir):


    # try:
    pdf_images = pdf2image.convert_from_path(os.path.join(data_dir, pdf_file))
    pdf_annot_images = copy.deepcopy(pdf_images)
    # except:
    #     return

    page_tokens = []
    txt_data = []
    layout_data = []
    # try:
    pdf = pdfplumber.open(os.path.join(data_dir, pdf_file))
    # except:
    #     return

    for page_id in tqdm(range(len(pdf.pages))):
        tokens = []
        src_word_bboxes = []
        filename = pdf_file.replace('.pdf', '')
        print(f'processing: {filename}, page_id: {page_id+1}')
        src_text = ''
        this_page = pdf.pages[page_id]
        width = int(this_page.width)
        height = int(this_page.height)
        anno_img = np.ones([int(this_page.width), int(this_page.height)] + [3], dtype=np.uint8) * 255
        anno_image = pdf_annot_images[page_id]
        img_width = anno_image.width
        img_height = anno_image.height


        words = this_page.extract_words(x_tolerance=1.5)

        lines = []
        for obj in this_page.layout._objs:
            if not isinstance(obj, LTLine):
                continue
            lines.append(obj)

        for i, word in enumerate(words):
            word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
            objs = []
            for obj in this_page.layout._objs:
                if not isinstance(obj, LTChar):
                    continue
                obj_bbox = (obj.bbox[0], float(this_page.height) - obj.bbox[3],
                            obj.bbox[2], float(this_page.height) - obj.bbox[1])
                if within_bbox(word_bbox, obj_bbox):
                    objs.append(obj)
            fontname = []
            for obj in objs:
                fontname.append(obj.fontname)
            if len(fontname) != 0:
                c = Counter(fontname)
                fontname, _ = c.most_common(1)[0]
            else:
                fontname = 'default'

            x0_, y0_, x1_, y1_ = int(word_bbox[0] / width * img_width), int(word_bbox[1] / height * img_height), int(
                word_bbox[2] / width * img_width), int(word_bbox[3] / height * img_height)
            # format word_bbox
            f_x0 = min(1000, max(0, int(word_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(word_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(word_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(word_bbox[3] / height * 1000)))
            word_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = word_bbox

            draw = ImageDraw.Draw(anno_image, "RGB")
            # get a font
            fnt = ImageFont.truetype("Chalkduster.ttf", 14)
            draw.text((x0_, y0_ - 15), str(i), font=fnt, fill='red')
            draw.rectangle((x0_, y0_, x1_, y1_), outline='red', width=3)

            # anno_color = [255, 0, 0]
            # for x in range(x0, x1):
            #     for y in range(y0, y1):
            #         anno_img[x, y] = anno_color
            if word['text'] != '':
                src_word_bboxes.append(word_bbox)
            word_bbox = tuple([str(t) for t in word_bbox])
            word_text = re.sub(r"\s+", "", word['text'])
            src_text += ' '+ word['text']
            tokens.append((word_text,) + word_bbox + (fontname,))

        src_text = src_text.strip()

        for figure in this_page.images:
            figure_bbox = (float(figure['x0']), float(figure['top']), float(figure['x1']), float(figure['bottom']))

            # format word_bbox
            f_x0 = min(1000, max(0, int(figure_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(figure_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(figure_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(figure_bbox[3] / height * 1000)))
            figure_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = figure_bbox
            x0, y0, x1, y1 = int(x0 * img_width / 1000), int(y0 * img_height / 1000), int(x1 * img_width / 1000), int(
                y1 * img_height / 1000)

            draw = ImageDraw.Draw(anno_image, "RGB")
            # get a font
            draw.rectangle((x0, y0, x1, y1), outline='blue', width=3)

            # anno_color = [0, 255, 0]
            # for x in range(x0, x1):
            #     for y in range(y0, y1):
            #         anno_img[x, y] = anno_color

            figure_bbox = tuple([str(t) for t in figure_bbox])
            word_text = '##LTFigure##'
            fontname = 'default'
            tokens.append((word_text,) + figure_bbox + (fontname,))

        for line in this_page.lines:
            line_bbox = (float(line['x0']), float(line['top']), float(line['x1']), float(line['bottom']))
            # format word_bbox
            f_x0 = min(1000, max(0, int(line_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(line_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(line_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(line_bbox[3] / height * 1000)))
            line_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = line_bbox
            x0, y0, x1, y1 = int(x0 * img_width / 1000), int(y0 * img_height / 1000), int(x1 * img_width / 1000), int(
                y1 * img_height / 1000)

            draw = ImageDraw.Draw(anno_image, "RGB")
            # get a font
            draw.rectangle((x0, y0, x1, y1), outline='green', width=3)

            # anno_color = [0, 0, 255]
            # for x in range(x0, x1 + 1):
            #     for y in range(y0, y1 + 1):
            #         anno_img[x, y] = anno_color

            line_bbox = tuple([str(t) for t in line_bbox])
            word_text = '##LTLine##'
            fontname = 'default'
            tokens.append((word_text,) + line_bbox + (fontname, ))

        tgt_text = src_text
        if src_text:
            reference = tgt_text.split(" ")
            hypothesis = src_text.split(" ")
        else:
            reference = []
            hypothesis = []

        tgt_idx = [i for i in range(len(src_word_bboxes))]
        tgt_word_bboxes = [src_word_bboxes[i] for i in tgt_idx]

        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

        # create ReadingBank dataset
        readingbank_txt = {
            "src": src_text,
            "tgt": tgt_text,
            "bleu": BLEUscore,
            "tgt_index": tgt_idx,
            "original_filename": filename,
            "filename": filename+'_page_'+str(page_id),
            "page_idx": page_id
        }

        readingbank_layout = {"src":src_word_bboxes ,"tgt": tgt_word_bboxes}

        # anno_img = np.swapaxes(anno_img, 0, 1)
        # anno_img = Image.fromarray(anno_img, mode='RGB')
        page_tokens.append((page_id, tokens, anno_img))

        isExist = os.path.exists(output_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(output_dir)

        pdf_images[page_id].save(
            os.path.join(output_dir, filename + '_{}_ori.jpg'.format(str(page_id))))
        anno_image.save(
            os.path.join(output_dir, filename + '_{}_ann.jpg'.format(str(page_id))))
        with open(os.path.join(output_dir, filename + '_{}.txt'.format(str(page_id))),'w',encoding='utf8') as fp:
            for token in tokens:
                fp.write('\t'.join(token) + '\n')

        txt_data.append(readingbank_txt)
        layout_data.append(readingbank_layout)

    with open(os.path.join(output_dir, filename + '_text.docparse_json'),'w',encoding='utf8') as fp:
        fp.write(ndjson.dumps(txt_data, ensure_ascii=False))

    with open(os.path.join(output_dir, filename + '_layout.docparse_json'),'w',encoding='utf8') as fp:
        fp.write(ndjson.dumps(layout_data,ensure_ascii = False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default='/Users/data/Documents/GitHub/pdf_parser/cv_pdf',
        type=str,
        required=False,
        help="The input data dir. Should contain the pdf files.",
    )
    parser.add_argument(
        "--output_dir",
        default='/Users/data/Documents/GitHub/pdf_parser/output',
        type=str,
        required=False,
        help="The output directory where the output data will be written.",
    )
    args = parser.parse_args()

    pdf_files = list(os.listdir(args.data_dir))
    pdf_files = [t for t in pdf_files if t.endswith('.pdf')]

    pool = multiprocessing.Pool(processes=1)
    print(pdf_files)
    for pdf_file in tqdm(pdf_files):
        pool.apply_async(worker, (pdf_file, args.data_dir, args.output_dir))
        worker(pdf_file, args.data_dir, args.output_dir)

    pool.close()
    pool.join()
