import pandas as pd
import glob
import os
import ndjson
from ast import literal_eval
import fitz
import random
import time

# To prepare dataset.docparse_json for training, you need to:
# 1. Prepare input pdf in 'pdf/'
# 2. Prepare ground truth labelled csv in 'csv/', with columns ['token','bbox','page_id','truth']
# 3. Prepare pdf images in 'img/' by running the preprocessing scripts pdf_process.py in the scripts directory
# annotation of ground truth on pdf is provided as optional function here

files = glob.glob('../../data/csv_gt/*_content_tokens_gt.csv')

def page_size(doc, page_id):
    w, h = int(doc[page_id].rect.width), int(doc[page_id].rect.height)
    return w, h

def denormalize_bbox(norm_bbox, width, height):
    norm_bbox = (float(norm_bbox[0]), float(norm_bbox[1]), float(norm_bbox[2]), float(norm_bbox[3]))
    return [int(norm_bbox[0] * width / 1000),
            int(norm_bbox[1] * height / 1000),
            int(norm_bbox[2] * width / 1000),
            int(norm_bbox[3] * height / 1000)]

def annot_pdf_page(doc, page_id, text_label, norm_bbox, color=None):
    if color is None:
        random.seed(time.process_time())
        color = (random.random(), random.random(), random.random())
    w, h = page_size(doc, page_id)
    doc[page_id].clean_contents()
    x0, y0, x1, y1 = denormalize_bbox(norm_bbox, w, h)
    doc[page_id].insert_text((x0, y0 - 2), text_label, fontsize=8, color=color)
    doc[page_id].draw_rect((x0, y0, x1, y1), color=color, width=1)
    return doc

style_color = {'abstract':(1,0,1), 'author':(0.5,0,0.5), 'caption':(0,1,0), 'date':(1,1,1), 'equation':(0.5,0.5,0.5), 'figure':(1,0.85,0),
               'footer':(0,0,1), 'list':(0,0.75,1), 'paragraph':(1,0,0), 'reference':(0.8,0.2,0), 'section':(0.2,0.8,0.2), 'table':(1,0.65,0), 'title':(0,0.5,0)}
datasets = []

for file in files:
    df = pd.read_csv(file)
    df = df.dropna(subset=['truth'])
    print(file)
    page_no = list(set(df['page_id'].values))
    fname = os.path.basename(file).split('.')[0].split('_content_tokens_gt')[0]
    docname = f'../../data/pdf/{fname}.pdf'
    doc = fitz.open(docname)
    for page_id in page_no:
        imgfile = './img/' + fname + '_'+ str(page_id) +'_ori.jpg'
        dataset = {}
        dataset['id'] = str(page_id)
        dataset['tokens'] = list(df.loc[df['page_id']==page_id,'token'].astype(str))
        dataset['bboxes'] = list(df.loc[df['page_id']==page_id,'bbox'].apply(literal_eval))
        dataset['ner_tags'] = list(df.loc[df['page_id']==page_id,'truth'].astype(str))
        dataset['image_path'] = imgfile

        datasets.append(dataset)

    # Visualize the ground truth labels on pdf document
    # for index, row in df.iterrows():
    #     # print(row['token'])
    #     page_id= int(row['page_id'])
    #     doc = annot_pdf_page(doc, page_id - 1, row['truth'], literal_eval(row['bbox']), color=style_color[row['truth']])
    # isExist = os.path.exists('annot_pdf_gt')
    # if not isExist:
    #     os.makedirs('annot_pdf_gt')
    # doc.save(f'annot_pdf_gt/{fname}_annot_ele_gt.pdf')

with open('datasets.docparse_json','w',encoding='utf8') as fp:
    fp.write(ndjson.dumps(datasets,ensure_ascii=False))