import os
import re
import csv
import json
import glob
import multiprocessing as mp
import time
import tqdm
import fitz
import camelot
import pandas as pd
import requests
import timeit
import nltk
import math
import ast
import datetime
from nltk.tokenize import sent_tokenize
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from postgresql_storage.db_util import DbUtil
from postgresql_storage.metric_extraction_models import MetricSchema
from postgresql_storage.reasoning_extraction_models import TargetAspect

from api.config import *
from general_document_parsing.document_parser import DocParser
from general_document_parsing.table_extraction import *

from general_table_extraction.detectron2.config.config import get_cfg
from general_table_extraction.demo.predictor import VisualizationDemo
from general_table_extraction.detectron2.data.detection_utils import read_image
from uie_tools.utils import *
from uie_tools.uom_conversion import *
from uie_tools.postprocessing import convert_unit_and_value, convert_str_to_float
from table_transformer.tsr import table_to_df
from table_transformer.core import TableDetector, TableRecognizer
import pytz
import traceback

mp.set_start_method("spawn", force=True)

KEY_ENV_REGEX = r'(ENVIRONMENTAL|ENVIRONMENTAL ASPECTS|ENVIRONMENTAL MANAGEMENT|ENVIRONMENTAL PROTECTION|EMISSIONS|Emission Management|key performance|performance data)'
SIM_MODEL = SentenceTransformer("models/checkpoints/all-MiniLM-L6-v2")


def log_task2csv(csv_name, row):
    '''
    log the record 'row' into path 'csv_name'
    '''
    import csv
    import os.path

    file_exists = os.path.isfile(csv_name)
    # Open the CSV file in "append" mode
    with open(csv_name, 'a', newline='') as f:
        # Create a dictionary writer with the dict keys as column fieldnames
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader() # file doesn't exist yet, write a header
            # Append single row to CSV
        writer.writerow(row)


def get_filepath_with_filename(parent_dir, filename):
    '''
    Given the target filename and a parent directory, traverse file system under the parent directory to lookup for the filename
    return the relative path to the file with filename if match is found
    otherwise raise a FileNotFoundError
    '''
    import errno

    for dirpath, subdirs, files in os.walk(parent_dir):
        for x in files:
            if x in filename:
                return os.path.join(dirpath, x)
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)


def document_parser(filename, use_model = USE_MODEL, do_annot = DOCPARSE_PDF_ANNOTATE, document_type = DOCPARSE_DOCUMENT_TYPE):
    '''
    input the filename (string of filename or list of filenames) that exists in 'data/pdf' to perform document parsing
    '''
    do_segment = True  # True for text block segmentation

    if isinstance(filename, str):
        inpaths = [get_filepath_with_filename(PDF_DIR,filename)]
    else:
        inpaths = [get_filepath_with_filename(PDF_DIR,i) for i in filename]
    for inpath in tqdm.tqdm(inpaths, desc='Doc Parsing'):
        start = timeit.default_timer()
        fname = os.path.basename(inpath)
        print("Parsing Report Name: ", fname)
        if document_type == 'esgReport':
            model_path = DOCPARSE_MODELV3_PATH
            sub_folder = 'esgReport/'
        elif document_type == 'agreement':
            model_path = DOCPARSE_MODELV3_CLAUSE_PATH
            key = ['TS', 'term sheet']
            sub_folder = 'FA/'
            if re.match(r'.*' + r'.*|.*'.join(key), fname, flags=re.IGNORECASE):
                model_path = DOCPARSE_MODELV3_TS_PATH
                document_type = 'termSheet'
                sub_folder = 'TS/'
        elif document_type == 'termSheet':
            model_path = DOCPARSE_MODELV3_TS_PATH
            key = ['FA', 'facility agreement', 'facilities agreement']
            sub_folder = 'TS/'
            if re.match(r'.*' + r'.*|.*'.join(key), fname, flags=re.IGNORECASE):
                model_path = DOCPARSE_MODELV3_CLAUSE_PATH
                document_type = 'agreement'
                sub_folder = 'FA/'
        parser = DocParser(
            inpath, DOCPARSE_OUTPUT_JSON_DIR + sub_folder, OUTPUT_TXT_DIR, OUTPUT_IMG_DIR, use_model, model_path ,OUTPUT_ANNOT_PDF_DIR, do_annot=do_annot, document_type=document_type
        )
        parser.process(do_segment)
        parser.save_output()
        if do_annot:
            parser.annot_pdf(do_segment)
        stop = timeit.default_timer()
        total_time = stop - start
        model_name = os.path.basename(model_path)
        row = {'task': 'Document Parsing',
               'filename': fname,
               'model_used': model_name,
               'num_pages': parser.doc.page_count,
               'num_tokens': len(parser.tokens),
               'inference_batch_size': DOCPARSE_BATCH_SIZE,
               'apply_model': parser.use_model,
               'runtime': total_time}
        log_task2csv(LOG_DIR + '/log_document_parsing.csv', row)

    print('document parser completed ...')


def get_parsed_doc(filename):
    ''' TODO: save parsed doc as output to evaluate model based parser #0829
    '''
    outpath = os.path.join(
        DOCPARSE_OUTPUT_JSON_DIR,
        re.sub('.pdf', '.json', filename)
    )
    with open(outpath, 'r') as f:
        output = json.load(f)
    heading_regex = '^title$|^section$|^title_\d{1,}$|^section_\d{1,}$'
    parsed_doc = []
    outpath = os.path.join(
        DOCPARSE_OUTPUT_JSON_DIR,
        re.sub('.pdf', '.json', filename)
    )
    with open(outpath, 'r') as f:
        output = json.load(f)
    for item in output['content']:
        for k, v in item.items():
            if re.search(heading_regex, k):
                dic = {}
                dic['title'] = item[k]
                dic['page_range'] = item['child_page_range'] if item['child_page_range'] else [item['page_id'],
                                                                                               item['page_id']]
                dic['filename'] = output['filename']
                if dic not in parsed_doc:
                    parsed_doc.append(dic)
    return parsed_doc


def find_env_related_parts(parsed_doc):
    env_pos = []
    page_range_list = []
    for item in parsed_doc:
        if item['page_range']:
            if isinstance(item['title'], str):
                # TODO: check parsed doc, some of titles are list
                if re.search(KEY_ENV_REGEX, item['title'], re.I):
                    if not re.search(NOT_KEY_ENV_REGEX, item['title'], re.I):
                        if item not in env_pos:
                            env_pos.append(item)
                        if item['page_range'] not in page_range_list:
                            page_range_list.append(item['page_range'])
    page_numbers = []
    for page_range in page_range_list:
        for p in range(page_range[0], page_range[1] + 1):
            page_numbers.append(p)
    page_numbers = list(set(page_numbers))
    return env_pos, page_numbers


def setup_cfg(args):
    # MODEL = SentenceTransformer("models/checkpoints/all-MiniLM-L6-v2")
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args['config_file'])
    cfg.merge_from_list(args['opts'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args['confidence_threshold']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args['confidence_threshold']
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args['confidence_threshold']
    cfg.freeze()
    return cfg


def get_predicted_img_coord(img_path):
    '''img_paths
    '''
    args = {
        'config_file': 'general_table_extraction/demo/All_X101.yml',
        'opts': ['MODEL.WEIGHTS', 'general_table_extraction/demo/model_final_X101.pth'],
        'input': [img_path],
        'output': OUTPUT_PRED_IMG_DIR,
        'confidence_threshold': 0.5
    }

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    img_info = {}
    for img_path in args['input']:
        input_ = glob.glob(os.path.expanduser(img_path))

        for path in tqdm.tqdm(input_, disable=not args['output']):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            out_filename = os.path.join(args['output'], os.path.basename(path))
            visualized_output.save(out_filename)

            instances = predictions['instances']
            boxes = instances.get_fields()['pred_boxes']
            if boxes:
                img_h, img_w = instances.image_size
                areas = list(boxes.tensor.detach().numpy())
                img_info = {
                    'img_path': img_path,
                    'box_area': areas,
                    'image_height': img_h,
                    'image_weight': img_w
                }
            print('{} completed in {}s'.format(path, time.time() - start_time))
    return img_info


def img_to_pdf_coord_conversion(img_box_coord, pdf_w, pdf_h, img_w, img_h):
    img_box_coord_transfer = [
        img_box_coord[0], img_h - img_box_coord[1],
        img_box_coord[2], img_h - img_box_coord[3]
    ]
    return [
        img_box_coord_transfer[0] * pdf_w / img_w,
        img_box_coord_transfer[1] * pdf_h / img_h,
        img_box_coord_transfer[2] * pdf_w / img_w,
        img_box_coord_transfer[3] * pdf_h / img_h,
    ]


def get_table_text_list(table_df, mode='TSR'):
    env_unit_df = pd.read_csv(METRIC_SCHEMA_CSV)
    metric_list = [ast.literal_eval(i) for i in env_unit_df.metric]
    metric_list = sorted(list(set(tuple(i) for i in metric_list)))
    if mode == 'TSR':
        df = table_df
        df = df.fillna('')
    else:
        # check row & column
        col_length = len(table_df.columns)
        i = 0
        for record in table_df.to_records(index=False):
            not_null_items = [item for item in record if item]
            i += 1
            # if len(not_null_items) > math.ceil(col_length/2):
            if len(not_null_items) > col_length / 2.:
                columns = list(record)
                break
        print(columns)

        df = pd.DataFrame(
            table_df[i:].to_numpy(),
            columns=columns
        )

    k = -1
    j = -1
    for col in df.columns:
        similar_pairs = get_similarity_sentbert(
            list(set(df[col])), metric_list,
            sim_threshold=0.7,
            pretrained=SIM_MODEL
        )
        m = len(similar_pairs)
        print(col, m)
        if m > k:
            k = m
            metric_col_name = col
            j += 1

    if k <= 0:
        # check if metrics in header
        header_similar_pairs = get_similarity_sentbert(
            list(df.columns), metric_list,
            sim_threshold=0.7,
            pretrained=SIM_MODEL
        )
        if len(header_similar_pairs) >= len(df.columns) / 2:
            # if metrics in header, transpose the dataframe
            print('tranpose dataframe ...')
            df_T = df.T
            df_T = df_T.reset_index()
            df_T = df_T.rename(columns=df_T.iloc[0]).drop(df_T.index[0])
            df = df_T
            k = -1
            j = -1
            for col in df.columns:
                similar_pairs = get_similarity_sentbert(
                    list(set(df[col])), metric_list,
                    sim_threshold=0.7,
                    pretrained=SIM_MODEL
                )
                m = len(similar_pairs)
                print(col, m)
                if m > k:
                    k = m
                    metric_col_name = col
                    j += 1

    table_text_list = []
    contents = df.values
    titles = df.columns
    for idx in range(0, len(contents)):
        content = contents[idx]
        # TODO: not just choose the only column as the metric column but max to 2 potential columns
        name = content[max(0, j)]
        if name:
            dic = dict()
            dic['name'] = name
            for k, v in zip(range(len(content)), content):
                if k != j:
                    dic.update({
                        titles[k].lower(): v
                    })
                    if dic not in table_text_list:
                        table_text_list.append(dic)
    return table_text_list


def update_metric_schema_from_csv(csv_path, db_table, tablename="metric_schema",
                                  filter_list=['metric']):
    du = DbUtil()
    # db_table.objects.all().delete() # erase all data in the existing schema table
    # import and update metric_schema schema from csv to target database table "metric_schema"
    reader = pd.read_csv(csv_path, encoding="unicode_escape", keep_default_na=False)
    columns = reader.columns.to_list()
    for id, row in reader.iterrows():
        row = row.values.tolist()
        e = list(zip(columns, row))
        dic = {}
        for k, v in e:
            if v == '' or pd.isna(v):
                v = None
            elif v in ['FALSE', 'false', 'N', 'n', 'No', 'no']:
                v = False
            elif v in ['TRUE', 'true', 'Y', 'y', 'Yes', 'yes']:
                v = True
            dic[k] = v
        if id != 0 and not db_table.objects.filter(metric=dic['metric']).filter(unit=dic['unit']).exists():
            db_table.objects.get_or_create(**dic)
        elif id != 0 and db_table.objects.filter(metric=dic['metric']).filter(unit=dic['unit']).exists():
            filter_dict = {k: v for k, v in dic.items() if k in filter_list}
            update_dict = {k: v for k, v in dic.items() if k not in filter_list}
            du.update_data(tablename, filter_dict=filter_dict, update_dict=update_dict)


def update_target_aspect_schema_from_csv(csv_path, db_table, tablename="target_aspect", filter_list=['target_aspect']):
    du = DbUtil()
    # db_table.objects.all().delete() # erase all data in the existing schema table
    # import and update target_aspect schema from csv to database table "target_aspect"
    with open(csv_path) as f2:
        reader2 = csv.reader(f2)
        for id, row in enumerate(reader2):
            if id == 0:
                columns = [i.encode("ascii", "ignore").decode() for i in row]
            e = list(zip(columns, row))
            dic = {}
            for k, v in e:
                if v == '' or pd.isna(v):
                    v = None
                elif v in ['FALSE', 'false', 'N', 'n', 'No', 'no']:
                    v = False
                elif v in ['TRUE', 'true', 'Y', 'y', 'Yes', 'yes']:
                    v = True
                dic[k] = v
            if id != 0 and not db_table.objects.filter(target_aspect=dic['target_aspect']).exists():
                db_table.objects.get_or_create(**dic)
            elif id != 0 and db_table.objects.filter(target_aspect=dic['target_aspect']).exists():
                filter_dict = {k: v for k, v in dic.items() if k in filter_list}
                update_dict = {k: v for k, v in dic.items() if k not in filter_list}
                du.update_data(tablename, filter_dict=filter_dict, update_dict=update_dict)


def extract_key_metrics(filename, table_text_list):
    ''' identity column type from raw dataframe
    column header format 1: metrics (metric name) + unit (unit value) + year (data value)
    column header format 2: metrics (metric name) + unit (unit value) + amount (data value)
    TODO:
    - metric name extraction: processed in get table text?
    - unit extraction: list the common types / position / format
    '''
    key_metrics_list = []
    for item in table_text_list:
        unit = ''
        for k in item.keys():
            # TODO: add config for unit pattern
            if re.search('unit', k) or re.search('单位', k) or re.search('單位', k):
                unit = item.get(k)
                break
        for k in item.keys():
            dic = dict()
            raw_value = ''
            dic['position'] = filename
            dic['company_name'] = filename.split('/')[-1].split('_')[0]
            dic['metric'] = item['name']
            dic['unit'] = unit
            if re.search('\(|\)', item['name']):
                name = re.split('\(|\)', item['name'])[0]
            else:
                name = item['name']
            year_re = '20\d\d'
            # TODO: classify the column to metric or data in "get table text" func
            # or add config for the data column name keywords
            other_quant_re = 'emissions|emission|amount|data|value'
            if re.search(year_re, k):
                dic['year'] = re.findall(year_re, k, re.I)[0]
                raw_value = item[k]
            elif re.search(other_quant_re, k, re.I):
                dic['year'] = re.findall(year_re, filename)[0]
                raw_value = item[k]
            if raw_value:
                # case 1: 100 kg
                if isinstance(raw_value, str):
                    if len(raw_value.split(' ')) > 1:
                        dic['value'] = raw_value
                        if dic['unit'] == '':
                            dic['unit'] = raw_value.split(' ')[-1]
                    # case 2: 100 (kg)
                    elif len([v for v in re.split('\(|\)', raw_value) if v]) > 1:
                        dic['value'] = [v for v in re.split('\(|\)', raw_value) if v][0]
                        if dic['unit'] == '':
                            dic['unit'] = [v for v in re.split('\(|\)', raw_value) if v][-1]
                    else:
                        dic['value'] = raw_value
            if dic['unit'] == '':
                regex = '\(|\)'
                if re.search(regex, dic['metric']):
                    dic['unit'] = [u for u in re.split(regex, dic['metric']) if u][-1]
            if dic.get('value'):
                if dic not in key_metrics_list:
                    key_metrics_list.append(dic)
    return key_metrics_list


def doc_parse_extract_metric(filename):
    inpath = get_filepath_with_filename(PDF_DIR, filename)
    outpath = os.path.join(
        DOCPARSE_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename)
    )
    if not os.path.exists(outpath):
        document_parser(filename)
    parsed_doc = get_parsed_doc(filename)
    env_pos = find_env_related_parts(parsed_doc)
    excel_filenames = save_possible_table_page(env_pos)
    table_text_list = extract_table_text(excel_filenames)
    key_metrics_list = extract_key_metrics(table_text_list)
    print('table extraction completed ...')
    return key_metrics_list


def map_metrics(key_metrics_list):
    env_unit_df = pd.read_csv(METRIC_SCHEMA_CSV, encoding='utf8')
    metric_list = [ast.literal_eval(i) for i in env_unit_df.metric]
    metric_list = sorted(list(set(tuple(i) for i in metric_list)))

    intensity_metric_unit = {}
    intensity_df = env_unit_df[env_unit_df.category == 'Intensity']
    intensity_metric_list = []

    for item in intensity_df.to_dict(orient='records'):
        metric = ast.literal_eval(item['metric'])
        unit = item['unit']
        metric_and_unit = [m + ' ' + unit for m in metric]
        intensity_metric_list.append(metric_and_unit)
        intensity_metric_unit[metric_and_unit[0]] = metric

    intensity_metric_list = sorted(
        list(
            set(tuple(i) for i in intensity_metric_list)
        )
    )

    def process_metric(metric):
        # TODO: need updated mapping method for upper level metric
        # *upper level column content*column content
        metric = re.sub('\*', ' ', metric).strip()
        metric = re.sub('<s>(.*?)</s>|\n|\((.*?)\)', '', metric)
        if re.search('\(|\)', metric):
            processed = re.split('\(|\)', metric)[0]
        else:
            processed = metric
        if metric.startswith('-'):
            processed = processed[1:]
        elif metric.endswith('-'):
            processed = processed[:-1]
        processed = processed.strip()
        return processed

    metric_mapper = {}
    names = list(set([item['metric'] for item in key_metrics_list]))
    for record in key_metrics_list:
        metric = record['metric']
        unit = record['unit']
        if re.search('intensity|per', metric, re.I) or re.search('/', unit, re.I):
            if re.search('RMB|HK|million|revenue', metric + unit, re.I):
                metric_mapper[metric] = process_metric(metric) + ' per revenue'
            elif re.search('volume|produced|MWh/1,000 tonnes|barrel|boe', metric + unit, re.I):
                metric_mapper[metric] = process_metric(metric) + ' per production'
            elif re.search('employee', metric + unit, re.I):
                metric_mapper[metric] = process_metric(metric) + ' per employee'
            else:
                metric_mapper[metric] = process_metric(metric + " " + record['unit'])
        else:
            metric_mapper[metric] = process_metric(metric)

    intensity_names = []
    not_intensity_names = []

    for name in names:
        if re.search('intensity', name, re.I):
            intensity_names.append(metric_mapper[name])
        else:
            not_intensity_names.append(metric_mapper[name])

    similar_pairs = {}
    intensity_similar_pairs = {}
    not_intensity_similar_pairs = {}

    if intensity_names:
        if intensity_metric_list:
            by_production_names = []
            by_revenue_names = []
            by_employee_names = []
            other_names = []
            for name in intensity_names:
                if re.search('per production', name, re.I):
                    by_production_names.append(name)
                elif re.search('per revenue', name, re.I):
                    by_revenue_names.append(name)
                elif re.search('per employee', name, re.I):
                    by_employee_names.append(name)
                else:
                    other_names.append(name)
            raw_intensity_similar_pairs = {}
            if by_production_names:
                pairs = get_similarity_sentbert(
                    by_production_names,
                    [m for m in intensity_metric_list if re.search('per production', m[0], re.I)],
                    sim_threshold=0
                )
                raw_intensity_similar_pairs.update(pairs)
            if by_employee_names:
                pairs = get_similarity_sentbert(
                    by_employee_names,
                    [m for m in intensity_metric_list if re.search('per employee', m[0], re.I)],
                    sim_threshold=0
                )
                raw_intensity_similar_pairs.update(pairs)
            if by_revenue_names:
                pairs = get_similarity_sentbert(
                    by_revenue_names,
                    [m for m in intensity_metric_list if re.search('per revenue', m[0], re.I)],
                    sim_threshold=0
                )
                raw_intensity_similar_pairs.update(pairs)
            if other_names:
                pairs = get_similarity_sentbert(
                    other_names,
                    intensity_metric_list,
                    sim_threshold=0
                )
                raw_intensity_similar_pairs.update(pairs)
            intensity_similar_pairs = {}
            for k, v in raw_intensity_similar_pairs.items():
                intensity_similar_pairs[k] = {
                    'similar_metrics': intensity_metric_unit[v['similar_metrics']][0],
                    'score': v['score']
                }
        else:
            intensity_similar_pairs = get_similarity_sentbert(
                intensity_names, metric_list,
                sim_threshold=0
            )
    if not_intensity_names:
        not_intensity_similar_pairs = get_similarity_sentbert(
            not_intensity_names, metric_list,
            sim_threshold=0
        )
    similar_pairs.update(intensity_similar_pairs)
    similar_pairs.update(not_intensity_similar_pairs)
    for item in key_metrics_list:
        item['target_metric'] = similar_pairs[metric_mapper[item['metric']]]['similar_metrics']
        item['similar_score'] = similar_pairs[metric_mapper[item['metric']]]['score']
    print('key_metrics_list: ', key_metrics_list)
    return key_metrics_list


from table_transformer.core import TableDetector
from table_transformer.utils import table_detection

DETECTION_THRESH = 0.8
DETECTION_KEYWORDS = ['emission', 'scope', 'energy', 'ghg', 'employee']


def find_table_info_by_page(tables, page_no):
    candidate = [tab for tab in tables if tab["page_no"] == page_no]
    if len(candidate) == 0:
        return None
    else:
        return candidate


def get_table_info(filename, page_numbers, model_det=None, mode='detectron2'):
    table_info = []
    if mode == 'detectron2':
        print('START: ', filename)
        fname = os.path.splitext(filename)[0]
        doc = fitz.open(get_filepath_with_filename(PDF_DIR, filename))
        page_count = doc.page_count
        img_count = len([file for file in os.listdir(OUTPUT_IMG_DIR) if re.search(fname + r'_\d+_ori.jpg', file)])

        if page_count != img_count:
            images = convert_from_path(get_filepath_with_filename(PDF_DIR, filename))

        for page_no in page_numbers:
            try:
                print('process page: ', page_no)
                page = doc[page_no - 1]
                pdf_w, pdf_h = page.rect.width, page.rect.height
                img_path = os.path.join(OUTPUT_IMG_DIR, f"{fname}_{page_no}_ori.jpg")
                if page_count != img_count:
                    img = images[page_no - 1]
                    if not os.path.exists(img_path):
                        img.save(img_path)
                img_info = get_predicted_img_coord(img_path)
                if img_info:
                    img_box_coord_list = img_info['box_area']
                    img_h = img_info['image_height']
                    img_w = img_info['image_weight']
                    i = 0
                    for img_box_coord in img_box_coord_list:
                        i += 1
                        pdf_box_coord = img_to_pdf_coord_conversion(
                            img_box_coord, pdf_w, pdf_h, img_w, img_h
                        )
                        table_areas = [
                            '{},{},{},{}'.format(
                                pdf_box_coord[0], pdf_box_coord[1],
                                pdf_box_coord[2], pdf_box_coord[3]
                            )
                        ]
                        table_info.append({
                            'filename': filename,
                            'page_no': page_no,
                            'table_areas': table_areas,
                            # 'records': table_df.to_dict(orient='records'),
                            # 'score': item['score']
                        })
            except Exception as e:
                traceback.print_exc()
                print(filename, page_no, e)
    elif mode == 'TSR':
        print('START: ', filename)
        fname = os.path.splitext(filename)[0]
        doc = fitz.open(get_filepath_with_filename(PDF_DIR, filename))

        if model_det is not None:
            model = model_det
        else:
            model = TableDetector(checkpoint_path="table_transformer/pubtables1m_detection_detr_r18.pth")
        raw_tables, _, _ = table_detection(
            doc, model, threshold=DETECTION_THRESH,
            # keywords=DETECTION_KEYWORDS,
            make_symmetric=False
        )

        for page_no in page_numbers:
            try:
                print('process page: ', page_no)
                img_info = find_table_info_by_page(raw_tables, page_no)
                if img_info:
                    i = 0
                    for tab in img_info:
                        i += 1
                        pdf_box_coord = tab["table_areas"]
                        table_areas = [pdf_box_coord]
                        table_info.append({
                            'filename': filename,
                            'page_no': page_no,
                            'table_areas': table_areas,
                            'score': tab["score"],
                            'enlarged_bbox': tab["enlarged_bbox"]
                        })
            except Exception as e:
                traceback.print_exc()
                print(filename, page_no, e)
    # save to json file
    # save_fname = f'table_info_{fname}.json'
    # with open('data/tables/{}'.format(save_fname), 'w') as f:
    #     json.dump(table_info, f, indent=4)
    return table_info


def main_func(filename, page_numbers, document_id=None, mode='TSR'):
    import torch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:3' if use_cuda else 'cpu')
    # year_re = '^f?y?2?0?\d\d'
    year_re = '^20\d\d'
    model = TableRecognizer(checkpoint_path="table_transformer/pubtables1m_structure_detr_r18.pth")
    table_info = get_table_info(filename, page_numbers, mode=mode)
    print('table_info: ', table_info)
    key_metrics_list = []
    if mode == 'TSR':
        for tab in table_info:
            opened_pdf = fitz.open(get_filepath_with_filename(PDF_DIR, filename))
            try:
                table_df = table_to_df(model, opened_pdf, tab)
                table_text_list = get_table_text_list(table_df, mode)
                print('table_text_list is completed: \n', table_text_list)
                key_metrics = extract_key_metrics(filename, table_text_list)
                print('key_metrics is completed: \n', key_metrics)
                if key_metrics:
                    key_metrics_n = []
                    for item in key_metrics:
                        item.update(
                            {
                                'page_no': tab['page_no'],
                                'table_area': tab['table_areas'][0],
                                'table_score': tab['score'],
                                'document_id': document_id
                            }
                        )
                        key_metrics_n.append(item)
                    key_metrics_list.extend(key_metrics_n)
            except:
                None
    else:
        for item in table_info:
            filename = item['filename']
            page_no = item['page_no']
            table_areas = item['table_areas']
            tables = camelot.read_pdf(
                filepath=get_filepath_with_filename(PDF_DIR, filename),
                pages=str(page_no),
                flavor='stream',
                row_tol=10,
                table_areas=table_areas,
                flag_size=True
            )
            if tables.n > 0:
                table_df = tables[0].df
                # process sub-level, this part can be removed under TSR
                raw_records = table_df.to_records(index=False)
                print(raw_records)
                records = [list(raw_records[0])]
                for idx in range(1, len(raw_records)):
                    e = raw_records[idx]
                    if e[0].startswith('–') or e[0].startswith('('):
                        new_item = [b + a for a, b in zip(e, records[-1])]
                        records = records[:-1]
                        records.append(new_item)
                    else:
                        records.append(list(e))
                table_df = pd.DataFrame(data=records)
                idx_list = []
                for idx in range(0, len(records)):
                    record = records[idx]
                    check = [1 if re.search(year_re, item) else 0 for item in record]
                    if sum(check) >= 2:
                        idx_list.append(idx)
                if len(idx_list) >= 2:
                    idx_list.extend([0, len(records) + 1])
                    idx_list = sorted(list(set(idx_list)))
                    table_text_list = []
                    for i in range(2, len(idx_list)):
                        if i == 2:
                            df = table_df.iloc[:idx_list[i], :]
                        else:
                            df = table_df.iloc[idx_list[i - 1]:idx_list[i], :]
                        sub_table_text_list = get_table_text_list(df, mode)
                        table_text_list.extend(sub_table_text_list)
                else:
                    table_text_list = get_table_text_list(table_df, mode)
                print('table_text_list is completed: \n', table_text_list)
                key_metrics = extract_key_metrics(filename, table_text_list)
                print('key_metrics is completed: \n', key_metrics)
                if key_metrics:
                    key_metrics_list.extend(key_metrics)
    if key_metrics_list:
        table_metric_df = pd.DataFrame(data=key_metrics_list)
        print('table metric df: ', table_metric_df.to_dict(orient='records'))
        results_updated = metrics_mapping(table_metric_df)
    else:
        results_updated = []
    print('results_updated', results_updated)
    return results_updated


def process_metric(metric):
    ''' clean metric
    '''
    if re.search('\(.*scope.*\)', metric, re.I):
        metric = u' '.join([item.strip() for item in re.split('\(|\)', metric)])

    metric = re.sub('<s>(.*?)</s>|\((.*?)\)|tons of carbon dioxide equivalent', '', metric)
    metric = re.sub('\n|\t', ' ', metric)
    if re.search('\(|\)', metric):
        processed = re.split('\(|\)', metric)[0]
    else:
        processed = metric
    if metric.startswith('-'):
        processed = processed[1:]
    elif metric.endswith('-'):
        processed = processed[:-1]
    processed = processed.strip()
    return processed


def process_unit(unit):
    if isinstance(unit, str):
        return re.sub('<s>|</s>|\n', '', unit)
    else:
        return unit


def gen_new_metric(metric, unit):
    if not isinstance(unit, str):
        unit = str(unit)
    if re.search('intensity|density| per', metric, re.I) or re.search('/|intensity| per', str(unit), re.I):
        if re.search('RMB|HK|million|revenue', metric + unit, re.I):
            if not re.search("MtCO2e", metric, re.I):
                new_metric = process_metric(metric) + ' per revenue'
            else:
                new_metric = process_metric(metric)
        elif re.search('volume|produced|MWh/1,000 tonnes|barrel|m2|square meter|unit|capita', metric + unit, re.I):
            new_metric = process_metric(metric) + ' per production'
        elif re.search('employee|person', metric + unit, re.I):
            new_metric = process_metric(metric) + ' per employee'
        else:
            new_metric = process_metric(metric + " " + unit)
    else:
        new_metric = process_metric(metric)
    return new_metric


def add_metric_label(metric):
    if not isinstance(metric, str):
        metric = str(metric)
    if not re.search('intensity|density| per', metric, re.I):
        label = 'not_intensity'
    else:
        if re.search('per production', metric, re.I):
            label = 'per_production_intensity'
        elif re.search('per revenue', metric, re.I):
            label = 'per_revenue_intensity'
        elif re.search('per employee', metric, re.I):
            label = 'per_employee_intensity'
        else:
            label = 'intensity'
    return label


def metrics_mapping(table_metric_df):
    print('*** metrics mapping starts: ')
    # env_unit_df = pd.read_csv(METRIC_SCHEMA_CSV)
    env_unit_df = pd.read_excel('data/schema/target_metric.xlsx')
    print('len of env_unit_df: ', len(env_unit_df))
    metric_list = [ast.literal_eval(i) for i in env_unit_df.metric]

    # todo: update this part
    metric_list = sorted(list(set(tuple(i) for i in metric_list)))

    intensity_metric_unit = dict()
    intensity_df = env_unit_df[env_unit_df.category == 'Intensity']
    intensity_metric_list = []

    for item in intensity_df.to_dict(orient='records'):
        metric = ast.literal_eval(item['metric'])
        unit = item['unit']
        metric_and_unit = [m + ' ' + unit for m in metric]
        intensity_metric_list.append(metric_and_unit)
        intensity_metric_unit[metric_and_unit[0]] = metric

    intensity_metric_list = sorted(
        list(
            set(tuple(i) for i in intensity_metric_list)
        )
    )

    prod_intensity_metric_list = [m for m in intensity_metric_list if re.search('per production', m[0], re.I)]
    ee_intensity_metric_list = [m for m in intensity_metric_list if re.search('per employee', m[0], re.I)]
    rev_intensity_metric_list = [m for m in intensity_metric_list if re.search('per revenue', m[0], re.I)]

    label_map = {
        'per_production_intensity': prod_intensity_metric_list,
        'per_revenue_intensity': rev_intensity_metric_list,
        'per_employee_intensity': ee_intensity_metric_list,
        'intensity': intensity_metric_list,
        'not_intensity': metric_list
    }

    table_metric_df['raw_value'] = table_metric_df['value']
    table_metric_df['raw_unit'] = table_metric_df['unit']
    table_metric_df['value'] = table_metric_df['raw_value'].apply(lambda i: convert_str_to_float(i))
    table_metric_df = table_metric_df[table_metric_df.value != '']
    if not table_metric_df.empty:
        table_metric_df['unit'] = table_metric_df['raw_unit'].apply(lambda i: process_unit(i))
        table_metric_df = table_metric_df.fillna('')
        table_metric_df['new_metric'] = table_metric_df.apply(
            lambda i: gen_new_metric(i['metric'], i['unit']),
            axis=1
        )
        table_metric_df['label'] = table_metric_df['new_metric'].apply(lambda i: add_metric_label(i))
        print('*** similar pairs starts: ')
        similar_pairs = {}
        for label, listname in label_map.items():
            print(label)
            metrics = list(set(table_metric_df[table_metric_df.label == label].new_metric))
            if metrics:
                raw_pairs = get_similarity_sentbert(
                    metrics,
                    listname,
                    sim_threshold=0,
                    pretrained=SIM_MODEL
                )
                if not label.startswith('not'):
                    pairs = dict()
                    for k, v in raw_pairs.items():
                        pairs[k] = {
                            'similar_metrics': intensity_metric_unit[v['similar_metrics']][0],
                            'score': v['score']
                        }
                    similar_pairs.update(pairs)
                else:
                    similar_pairs.update(raw_pairs)

        table_metric_df['target_metric'] = table_metric_df['new_metric'].apply(
            lambda i: similar_pairs[i]['similar_metrics']
        )

        table_metric_df['similar_score'] = table_metric_df['new_metric'].apply(
            lambda i: similar_pairs[i]['score']
        )
        table_metric_df_sorted = table_metric_df.sort_values(
            ['company_name', 'target_metric', 'similar_score', 'value', 'year', 'unit'],
            ascending=False
        )
        results_updated = table_metric_df_sorted.drop_duplicates(
            subset=[
                'company_name',
                'target_metric',
                'year',
            ]
        ).to_dict(orient='records')
    else:
        results_updated = []
    return results_updated


def detect_table_areas(filename, page_numbers):
    img_info_list = []
    print('START: ', filename)
    fname = os.path.splitext(filename)[0]
    doc = fitz.open(get_filepath_with_filename(PDF_DIR, filename))
    page_count = doc.page_count
    img_count = len([file for file in os.listdir(OUTPUT_IMG_DIR) if re.search(fname + r'_\d+_ori.jpg', file)])
    if page_count != img_count:
        images = convert_from_path(get_filepath_with_filename(PDF_DIR, filename))

    for page_no in page_numbers:
        page = doc[page_no - 1]
        pdf_w, pdf_h = page.rect.width, page.rect.height
        img_path = os.path.join(OUTPUT_IMG_DIR, f"{fname}_{page_no}_ori.jpg")
        if page_count != img_count:
            img = images[page_no - 1]
            if not os.path.exists(img_path):
                img.save(img_path)
        img_info = get_predicted_img_coord(img_path)
        if img_info:
            img_box_coord_list = img_info['box_area']
            img_h = img_info['image_height']
            img_w = img_info['image_weight']
            i = 0
            for img_box_coord in img_box_coord_list:
                i += 1
                pdf_box_coord = img_to_pdf_coord_conversion(
                    img_box_coord, pdf_w, pdf_h, img_w, img_h
                )
                table_areas = [
                    '{},{},{},{}'.format(
                        pdf_box_coord[0], pdf_box_coord[1],
                        pdf_box_coord[2], pdf_box_coord[3]
                    )
                ]
                img_info_list.append({
                    'filename': filename,
                    'page_no': page_no,
                    'i': i,
                    'table_areas': table_areas
                })
    return img_info_list


def concat_pattern(tag_list):
    s = ''
    for i, tag in enumerate(tag_list):
        if i < len(tag_list) - 1:
            s += tag + r'(_){0,1}\d{0,}$|'
        else:
            s += tag + r'(_){0,1}\d{0,}$'
    return s.strip()


def update_mer(filename):
    du = DbUtil()
    update_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
    inpath = os.path.join(
        METRIC_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename)
    )
    metric_list = [ast.literal_eval(i['metric']) for i in
                   du.select_table(table_name="metric_schema", field_list=['metric'])]
    metric_list = sorted(list(set(tuple(i) for i in metric_list)))

    with open(inpath, 'r') as f:
        data = json.load(f)

    result_metrics = []
    # append all metric into a list for target metric mapping
    # create a dictionary of position that map list id (k) to metric_entity_relations id (i,j), where i is result id, j is metric id, k is the list order id
    for id1, d1 in enumerate(data['metric_entity_relations']):
        for k1, v1 in d1.items():
            if type(v1) is list and any(i in k1 for i in TARGET_ELEMENTS):
                for id2, d2 in enumerate(v1):
                    for id3, d3 in enumerate(d2['ners']):
                        metric = d3["metric"]
                        number = d3["number"]
                        if metric in number or number in metric:  # do not consider metric or number which is a substring of another
                            continue
                        result_metrics.append(metric)

    # check if there is any valid result metrics. If not, the metric_target_metric_dict is empty
    if result_metrics:
        metric_target_metric_dict = get_similarity_sentbert(result_metrics, metric_list)
    else:
        metric_target_metric_dict = {}

    year = data["year"]
    company_name = data["company_name"]
    for id1, d1 in enumerate(data['metric_entity_relations']):
        page_id = d1["page_id"]
        text_block_id = d1["text_block_id"]
        for k1, v1 in d1.items():
            if type(v1) is list and any(i in k1 for i in TARGET_ELEMENTS):
                element = k1
                for id2, d2 in enumerate(v1):
                    # print('page_id: ',page_id,' element: ',k1,' item :',d2)
                    sent_id = d2['sent_id']
                    sentence = d2['sentence']
                    try:
                        metric_year = re.search(r'\b(19|20)\d{2}\b', sentence)[0]
                    except:
                        metric_year = year
                    for id3, d3 in enumerate(d2['ners']):
                        subject = target_aspect = disclosure \
                            = target_metric = sim_score \
                            = compulsory = intensity_group = converted_value \
                            = converted_unit = target_unit = multiplier = None
                        ner = data['metric_entity_relations'][id1][element][id2]["ners"][id3]
                        metric = d3["metric"]
                        number = d3["number"]
                        # target_metric = d3['target_metric']
                        relation, value, unit = get_relation_numeric_unit(number)
                        # target_metric, sim_score = get_similarity_sentbert(metric, metric_list)
                        if metric in metric_target_metric_dict:
                            target_metric = metric_target_metric_dict[metric]['similar_metrics']
                            sim_score = metric_target_metric_dict[metric]['score']
                        else:
                            sim_score = 0
                        if target_metric:
                            subject = MetricSchema.objects.filter(
                                metric__icontains=target_metric).values('subject').first()['subject']
                            target_aspect = MetricSchema.objects.filter(
                                metric__icontains=target_metric).values('target_aspect').first()['target_aspect']
                            disclosure = MetricSchema.objects.filter(
                                metric__icontains=target_metric).values('disclosure').first()['disclosure']
                            target_unit = MetricSchema.objects.filter(
                                metric__icontains=target_metric).values('unit').first()['unit']
                            compulsory = MetricSchema.objects.filter(
                                metric__icontains=target_metric).values('compulsory').first()['compulsory']
                            intensity_group = MetricSchema.objects.filter(
                                metric__icontains=target_metric).values('intensity_group').first()['intensity_group']
                            multiplier, converted_unit = convert_arbitrary_uom(unit, target_metric, metric_year)
                            if multiplier:
                                try:
                                    converted_value = float(value) * float(multiplier)
                                except:
                                    pass

                        filter_dict = {
                            'company_name': company_name,
                            'year': year,
                            'page_id': page_id,
                            'text_block_id': text_block_id,
                            'block_element': element,
                            'sentence': sentence,
                            'sent_id': sent_id,
                            'metric': metric,
                            'number': number
                        }
                        update_dict = {
                            'metric_year': metric_year,
                            'subject': subject,
                            'target_aspect': target_aspect,
                            'disclosure': disclosure,
                            'target_metric': target_metric,
                            'compulsory': compulsory,
                            'intensity_group': intensity_group,
                            'similarity': float(sim_score),
                            'relation': relation,
                            'original_value': value,
                            'unit': unit,
                            'uom_conversion_multiplier': multiplier,
                            'converted_value': converted_value,
                            'converted_unit': converted_unit,
                            'target_unit': target_unit,
                            'update_datetime': update_datetime
                        }
                        du.update_data(METRIC_EXTRACTION_TABLE_NAME, filter_dict=filter_dict, update_dict=update_dict)
                        update_dict['update_datetime'] = update_datetime.strftime("%d/%m/%Y, %H:%M:%S")
                        ner.update(update_dict)
    with open(inpath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data


def update_rer(filename):
    du = DbUtil()
    update_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
    inpath = os.path.join(
        REASONING_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename)
    )
    subject = [i['subject'] for i in
               du.select_table(table_name="metric_schema", field_list=['subject'])]
    target_aspect = [i['target_aspect'] for i in
                     du.select_table(table_name="metric_schema", field_list=['target_aspect'])]
    disclosure = [i['disclosure'] for i in
                  du.select_table(table_name="metric_schema", field_list=['disclosure'])]
    metric_list = [ast.literal_eval(i['metric']) for i in
                   du.select_table(table_name="metric_schema", field_list=['metric'])]
    metric_list2 = metric_list.copy()
    metric_list = sorted(list(set(tuple(i) for i in metric_list)))
    disclosure_kw_dict = {}
    disclosure_kw = list(zip(disclosure, metric_list2))
    for dis, kw in disclosure_kw:
        if dis not in disclosure_kw_dict:
            if isinstance(kw, list):
                disclosure_kw_dict[dis] = kw
            else:
                disclosure_kw_dict[dis] = []
                disclosure_kw_dict[dis].append(kw)
        else:
            if isinstance(kw, list):
                disclosure_kw_dict[dis].extend(kw)
            else:
                disclosure_kw_dict[dis].append(kw)

    disclosure_tgt_aspect = dict(zip(disclosure, target_aspect))
    disclosure_subject = dict(zip(disclosure, subject))

    with open(inpath, 'r') as f:
        data = json.load(f)
    result_targets = []
    for id1, d1 in enumerate(data['reasoning_entity_relations']):
        for k1, v1 in d1.items():
            if type(v1) is list:
                for id2, d2 in enumerate(v1):
                    for id3, d3 in enumerate(d2['ners']):
                        head_entity = d3["head_entity"]
                        head_entity_type = d3["head_entity_type"]
                        tail_entity = d3["tail_entity"]
                        tail_entity_type = d3["tail_entity_type"]
                        relation = d3["relation"]
                        is_valid_pair = uie_reasoning_validator(head_entity_type, head_entity, tail_entity_type,
                                                                tail_entity, relation)
                        if is_valid_pair:
                            result_targets.append(head_entity)

    # check if there is any valid result targets. If not, the target_kw_dict is empty
    if result_targets:
        target_kw_dict = get_similarity_sentbert(result_targets, metric_list, return_single_response=False)
    else:
        target_kw_dict = {}

    del_elements = {}
    year = data["year"]
    company_name = data["company_name"]
    for id1, d1 in enumerate(data['reasoning_entity_relations']):
        page_id = d1["page_id"]
        text_block_id = d1["text_block_id"]
        for k1, v1 in d1.items():
            if type(v1) is list:
                element = k1
                if not v1:
                    del_elements[id1] = element
                    continue
                for id2, d2 in enumerate(v1):
                    if not d2:
                        data['reasoning_entity_relations'][id1][element].remove(d2)
                        continue
                    sent_id = d2['sent_id']
                    sentence = d2['sentence']
                    for id3, d3 in enumerate(d2['ners']):
                        ner = data['reasoning_entity_relations'][id1][element][id2]["ners"][id3]
                        head_entity = d3["head_entity"]
                        head_entity_type = d3["head_entity_type"]
                        relation = d3["relation"]
                        tail_entity = d3["tail_entity"]
                        tail_entity_type = d3["tail_entity_type"]
                        subject = target_aspect = disclosure = relavent_keyword = sim_score = None
                        if head_entity_type == 'target':
                            if head_entity in target_kw_dict:
                                relavent_keyword = target_kw_dict[head_entity]['similar_metrics']
                                sim_score = target_kw_dict[head_entity]['score']
                            else:
                                sim_score = 0
                            # relavent_keyword, sim_score = get_similarity_sentbert(head_entity, metric_list, return_single_response=False)
                            if relavent_keyword:
                                if isinstance(relavent_keyword, str):
                                    for k, v in disclosure_kw_dict.items():
                                        if relavent_keyword in v:
                                            disclosure = k
                                            target_aspect = disclosure_tgt_aspect[k]
                                            subject = disclosure_subject[k]
                                            break
                                elif isinstance(relavent_keyword, list):
                                    if len(relavent_keyword) > 3:
                                        relavent_keyword = relavent_keyword[:3]
                                        sim_score = sim_score[:3]
                                    disclosure = []
                                    target_aspect = []
                                    subject = []
                                    for k, v in disclosure_kw_dict.items():
                                        for kw in relavent_keyword:
                                            if kw in v:
                                                disclosure.append(k)
                                                target_aspect.append(disclosure_tgt_aspect[k])
                                                subject.append(disclosure_subject[k])
                                                break
                        filter_dict = {
                            'company_name': company_name,
                            'year': year,
                            'page_id': page_id,
                            'text_block_id': text_block_id,
                            'block_element': element,
                            'sentence': sentence,
                            'sent_id': sent_id,
                            'head_entity': head_entity,
                            'tail_entity': tail_entity
                        }
                        update_dict = {
                            'subject': subject,
                            'target_aspect': target_aspect,
                            'disclosure': disclosure,
                            'target_metric': relavent_keyword,
                            'similarity': sim_score,
                            'update_datetime': update_datetime
                        }
                        du.update_data(REASONING_EXTRACTION_TABLE_NAME, filter_dict=filter_dict,
                                       update_dict=update_dict)
                        update_dict['update_datetime'] = update_datetime.strftime("%d/%m/%Y, %H:%M:%S")
                        ner.update(update_dict)
                    for id4, d4 in enumerate(d2['split_sentence']):
                        if len(d4["char_position"]) == 1:
                            data['reasoning_entity_relations'][id1][element][id2]["split_sentence"][id4][
                                "char_position"] = d4["char_position"][0]
    for k, v in del_elements.items():
        del data['reasoning_entity_relations'][k]
    with open(inpath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data

def init_UIE_model(model_name):
    from models.deeplearning.uie_model.uie_predictor import init_model
    model = init_model(model_name)
    time.sleep(3)
    return model

def delete_UIE_model():
    import gc
    if 'metrics_model' in globals():
        del metrics_model
    if 'reasoning_model' in globals():
        del reasoning_model
    gc.collect()

def metrics_predictor_predict(sent_list):
    if 'metrics_model' not in globals():
        metrics_model = init_UIE_model("metrics_model_v1")
    metrics_model_prediction = metrics_model.compute_prediction(sent_list)
    # metrics_model_prediction = [dict(i) for i in metrics_model_prediction if isinstance(i, dict)]
    return metrics_model_prediction

def reasoning_predictor_predict(sent_list):
    if 'reasoning_model' not in globals():
        reasoning_model = init_UIE_model("reasoning_model_v1")
    reasoning_model_prediction = reasoning_model.compute_prediction(sent_list)
    # reasoning_model_prediction = [dict(i) for i in reasoning_model_prediction if isinstance(i, dict)]
    return reasoning_model_prediction

def get_entity_relation(filename, url_metric, url_reasoning, document_id=None, model_version='v1',
                        do_metric_extract=True, do_reasoning_extract=True):
    # check if model for tokenizer exists
    try:
        nltk.data.find('punkt.zip')
    except LookupError:
        nltk.download('punkt')
    du = DbUtil()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import gc
    import language_tool_python

    tool = language_tool_python.LanguageTool(
        'en-US')  # use a LanguageTool local server (automatically set up), language English

    # target = ['caption', 'list', 'paragraph', 'section', 'table', 'title']
    # import and update metric schema from csv to database table "metric_schema"
    try:
        update_metric_schema_from_csv(METRIC_SCHEMA_CSV, MetricSchema, tablename="metric_schema",
                                      filter_list=['metric', 'unit'])
    except:
        pass

    # extract list of metrics from database table "environment metric unit"
    distinct_target_aspect = set([i['target_aspect'] for i in du.select_table(
        table_name="metric_schema",
        field_list=['target_aspect']
        # filter_dict={'subject_no': 'A'}
    )])
    distinct_disclosure = set([i['disclosure'] for i in du.select_table(
        table_name="metric_schema",
        field_list=['disclosure']
        # filter_dict={'subject_no': 'A'}
    )])
    metric_list = [ast.literal_eval(i['metric']) for i in
                   du.select_table(table_name="metric_schema", field_list=['metric'])]
    metric_list2 = metric_list.copy()
    metric_list = sorted(list(set(tuple(i) for i in metric_list)))
    flat_metric = set([item for sublist in metric_list for item in sublist])
    env_keywords = r'\b(' + r'|'.join(distinct_target_aspect) + \
                   r'|'.join(distinct_disclosure) + r'|'.join(flat_metric) + r')\b'
    # print(f'env_keywords = {env_keywords}')

    # extract list of target aspect from database table "target aspect"

    subject = [i['subject'] for i in
               du.select_table(table_name="metric_schema", field_list=['subject'])]
    target_aspect = [i['target_aspect'] for i in
                     du.select_table(table_name="metric_schema", field_list=['target_aspect'])]
    disclosure = [i['disclosure'] for i in
                  du.select_table(table_name="metric_schema", field_list=['disclosure'])]
    disclosure_kw_dict = {}
    disclosure_kw = list(zip(disclosure, metric_list2))
    for dis, kw in disclosure_kw:
        if dis not in disclosure_kw_dict:
            if isinstance(kw, list):
                disclosure_kw_dict[dis] = kw
            else:
                disclosure_kw_dict[dis] = []
                disclosure_kw_dict[dis].append(kw)
        else:
            if isinstance(kw, list):
                disclosure_kw_dict[dis].extend(kw)
            else:
                disclosure_kw_dict[dis].append(kw)

    disclosure_tgt_aspect = dict(zip(disclosure, target_aspect))
    disclosure_subject = dict(zip(disclosure, subject))

    inpath = os.path.join(
        DOCPARSE_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename)
    )
    if not os.path.exists(inpath):
        document_parser(filename)
    with open(inpath, "r") as f:
        parsed_pdf = json.load(f)
    filename = parsed_pdf['filename']
    if filename[0].isdigit():
        stock_id = filename.split('/')[-1].split('_')[0]
        company_name = filename.split('/')[-1].split('_')[1]
        year = min([int(i) for i in re.findall('\D(20\d{2})\D?', filename, re.I)])
    else:
        company_name = filename.split('/')[-1].split('_')[0]
        year = min([int(i) for i in re.findall(
            '\d+', filename, re.I) if re.match('\d{4}$', i)])
    document_id = company_name + '_' + str(year) + '_eng'
    sentences = []

    searched_text = []
    # extract text from parsed documents with element tags in ['paragraph','list','caption']
    for item in parsed_pdf['content']:
        page_id = item['page_id']
        text_block_id = item['id']
        for key, value in item.items():
            # if not re.match(r'title(_){0,1}\d{0,}$|section(_){0,1}\d{0,}$', key):
            #     continue
            # else:
            #     if re.search(env_keywords, value, re.I) and not re.search(NOT_KEY_ENV_REGEX, value, re.I) and item["child_content"]:
            #         print(f'Title or section with text: {value}')
            #         print(f'Matched environment keyword = {re.search(env_keywords, value, re.I)[1]} \n')
            #         for i in item["child_content"]:
            #             for key2, value2 in i.items():
            pattern = '|'.join([f'{i}' + r'(_){0,1}\d{0,}$' for i in TARGET_ELEMENTS])
            if re.match(pattern, key):
                # add flag isMatchedKeyword to check if the text contain target metric keywords
                if re.search(env_keywords, value, re.I) and not re.search(NOT_KEY_ENV_REGEX, value,
                                                                          re.I) and value not in searched_text:
                    isMatchedKeyword = True
                else:
                    isMatchedKeyword = False
                searched_text.append(value)
                new_key = key
                if sentences and key in sentences[-1] and page_id == sentences[-1]['page_id']:
                    key_tup = key.split('_')
                    if len(key_tup) == 2:
                        new_key = key_tup[0] + '_' + str(int(key_tup[1]) + 1)
                    else:
                        new_key = key + '_' + str(1)
                # tokenize the text into sentences
                sents = sent_tokenize(value)
                for i, sent in enumerate(sents):
                    # for each sentence, check if it contains Chinese characters or grammatically correct. Discard sentence if it is true
                    hasChinese = re.search(u'[\u4e00-\u9fff]', sent)
                    if hasChinese:
                        continue
                    if is_grammarly_incorrect(tool, sent):
                        # print(f'sentence with grammartical errors: {sent} \n')
                        continue
                    sentences.append({'company_name': company_name,
                                      'year': year,
                                      'page_id': page_id,
                                      'text_block_id': text_block_id,
                                      'block_element': new_key,
                                      'sent_id': i,
                                      'sentence': sent,
                                      'isMatchedKeyword': isMatchedKeyword,
                                      'document_id': document_id})
    del tool
    gc.collect()  # garbage collector at the beginning to explicitly free memory
    search_count = len(sentences)
    # print(f'no of sentences with matched keywords on target metric list: {search_count}')
    batch_num = math.ceil(search_count / UIE_BATCH_SIZE)
    filename_dict = {'filename': filename + '.pdf'}
    global metric_extract_time, reasoning_extract_time, total_metric_extract_time, total_reasoning_extract_time, mer_sent_count, total_mer_count, rer_sent_count, total_rer_count

    metric_extract_time = reasoning_extract_time = mer_sent_count = total_mer_count = rer_sent_count = total_rer_count = 0

    def text_metric_extract():
        global metric_extract_time, total_metric_extract_time, mer_sent_count, total_mer_count, metrics_model
        '''Batch metric entity relation extraction'''
        all_metrics_predictions = []
        t = tqdm.tqdm(range(batch_num), desc='Text Metric Extraction')
        metrics_model = init_UIE_model("metrics_model_v1")
        all_start1 = timeit.default_timer()
        for batch_counter1 in t:
            record_progress_update(t, PDFFILES_TABLENAME, filename_dict, 'process2_progress',
                                   'Text metrics extraction is processing ... ', 0.05)
            start1 = timeit.default_timer()
            batch1 = [i['sentence'] for i in
                      sentences[batch_counter1 * UIE_BATCH_SIZE:(batch_counter1 + 1) * UIE_BATCH_SIZE]]
            metrics_predictions = metrics_predictor_predict(batch1)

            # response1 = requests.post(url_metric, json={
            #     'input': [i['sentence'] for i in sentences[batch_counter1 * UIE_BATCH_SIZE:(batch_counter1 + 1) * UIE_BATCH_SIZE]]})
            # metrics_predictions = response1.json()

            stop1 = timeit.default_timer()
            metric_extract_time += stop1 - start1

            all_metrics_predictions.extend(metrics_predictions)
        del metrics_model
        gc.collect()

        prev_page_id = prev_text_block_id = prev_block_element = None
        metric_dic = []
        ent_data = []
        result_metrics = []
        metric_nested_dic = {'filename': filename, 'document_id': document_id, 'company_name': company_name,
                             'year': year,
                             'metric_entity_relations': []}

        # append all metric into a list for target metric mapping
        # create a dictionary of position that map list id (k) to metric_entity_relations id (i,j), where i is result id, j is metric id, k is the list order id
        for i, res1 in enumerate(all_metrics_predictions):
            # check if res1 is a dictionary and contain key 'relation'
            if isinstance(res1, dict) and 'relation' in res1:
                # check if res1['relation'] is a dictionary and contain key 'string'
                if isinstance(res1['relation'], dict) and 'string' in res1['relation']:
                    '''
                    if res1['relation']['string'] exists, it has structure:
                    "relation": {
                        "string": [
                            <RELATION>,
                            <HEAD_ENTITY_TYPE>,
                            <HEAD_ENTITY_STRING>,
                            <TAIL_ENTITY_TYPE>,
                            <TAIL_ENTITY_STRING>
                            ],...
                        }
                    '''
                    metric_entity_relations = res1['relation']['string']
                    for j, ent_rel in enumerate(metric_entity_relations):
                        relation, head_entity_type, head_entity, tail_entity_type, tail_entity = ent_rel
                        if head_entity_type == "metric":
                            metric = head_entity
                            number = tail_entity
                            if metric in number or number in metric:  # do not consider metric or number which is a substring of another
                                continue
                            result_metrics.append(metric)
        # print(f'valid result metrics = {result_metrics}')
        '''
        Generate the dictionary of metric to sim_score and target metric by input list of result metrics and list of list of target metric keywords
        Example output of metric_target_metric_dict:
        {
            "CO2 emissions": {
                "score": 0.7978004217147827,
                "similar_metrics": "Carbon Dioxides"
            },...
        }
        '''
        # check if there is any valid result metrics. If not, the metric_target_metric_dict is empty
        if result_metrics:
            keywords_count = sum([len(i) for i in metric_list])
            start_map1 = timeit.default_timer()
            metric_target_metric_dict = get_similarity_sentbert(result_metrics, metric_list)
            stop_map1 = timeit.default_timer()
            total_time_map1 = stop_map1 - start_map1
            row = {'task': 'Metric Mapping', 'filename': filename, 'num_result': len(result_metrics), 'runtime': total_time_map1}
            log_task2csv('data/log/log_target_metric_mapping.csv', row)
        else:
            metric_target_metric_dict = {}

        for i, res1 in enumerate(all_metrics_predictions):
            # check if res1 is a dictionary and contain key 'relation'
            if isinstance(res1, dict) and 'relation' in res1:
                # check if res1['relation'] is a dictionary and contain key 'string'
                if isinstance(res1['relation'], dict) and 'string' in res1['relation']:
                    '''
                    if res1['relation']['string'] exists, it has structure:
                    "relation": {
                        "string": [
                            <RELATION>,
                            <HEAD_ENTITY_TYPE>,
                            <HEAD_ENTITY_STRING>,
                            <TAIL_ENTITY_TYPE>,
                            <TAIL_ENTITY_STRING>
                            ],...
                        }
                    '''
                    metric_entity_relations = res1['relation']['string']
                else:
                    # print(
                    #     f'Non-dictionary Response with res1["relation"]={res1["relation"]} and dtype={type(res1["relation"])}')
                    continue
            else:
                # print(
                #     f'Non-dictionary Response with res1={res1} and dtype={type(res1)}')
                continue

            flat_list = [item for sublist in metric_entity_relations for item in sublist]
            metric_entity = res1['entity']['string']
            page_id = sentences[i]['page_id']
            text_block_id = sentences[i]['text_block_id']
            block_element = sentences[i]['block_element']
            sentence = sentences[i]['sentence']
            sent = {}
            value_unit = {'values': [], 'units': []}
            for ent in metric_entity:
                entity_type, entity = ent
                if entity_type == 'value':
                    value_unit['values'].append(entity)
                elif entity_type == 'unit':
                    value_unit['units'].append(entity)

            # print(f'Processing metric sentence no. {i}:\n {sentence}', '\n')
            # print(f'↖ With page_id = {page_id}, text_block_id = {text_block_id}, block_element = {block_element} \n')

            if metric_entity_relations:
                ners = []
                mer_sent_count += 1
                mer_count = 0
                entities = []
                split_sent = []
                searched = []
                same_element = block_element == prev_block_element
                same_block = text_block_id == prev_text_block_id
                same_page = page_id == prev_page_id

                # print('metric entity relation found: \n', json.dumps(metric_entity_relations, indent=4, sort_keys=True, default=str), '\n')
                for j, ent_rel in enumerate(metric_entity_relations):
                    relation, head_entity_type, head_entity, tail_entity_type, tail_entity = ent_rel
                    if head_entity_type == "metric":
                        metric = head_entity
                        number = tail_entity
                        if metric in number or number in metric:  # do not consider metric or number which is a substring of another
                            continue
                        isNumeric = has_numbers(
                            number)  # check if number has digit, if not it is not valid entity-relation pair
                        if isNumeric:
                            total_mer_count += 1
                            mer_count += 1
                            relation, value, unit = get_relation_numeric_unit(number)
                            relatives = {r'increas(.*)by': 'increased by',
                                         r'decreas(.*)by|reduc(.*)by|drop(.*)by|down(.*)by': 'decreased by'}
                            subject = target_aspect = disclosure = target_metric = sim_score \
                                = converted_value = converted_unit = multiplier \
                                = target_unit = compulsory = intensity_group = None
                            for k, v in relatives.items():
                                if re.search(k, sentence):
                                    relation = v
                                    break
                            if model_version != "v1":
                                for i, v in enumerate(value_unit['values']):
                                    if v in number:
                                        value = float(v)
                                        unit = value_unit['units'][i]
                                        break
                            else:
                                if isinstance(value, str) and len(value.split('-')) > 1:
                                    value = [float(value.split('-')[0]),
                                             float(value.split('-')[1])]
                            try:
                                metric_year = re.search(
                                    r'\b(19|20)\d{2}\b', sentence)[0]
                            except:
                                metric_year = year
                            # target_metric, sim_score = get_similarity_sentbert(metric, metric_list)
                            if metric in metric_target_metric_dict:
                                target_metric = metric_target_metric_dict[metric]['similar_metrics']
                                sim_score = metric_target_metric_dict[metric]['score']
                            else:
                                sim_score = 0
                            if target_metric:
                                subject = MetricSchema.objects.filter(
                                    metric__icontains=target_metric).values('subject').first()['subject']
                                target_aspect = MetricSchema.objects.filter(
                                    metric__icontains=target_metric).values('target_aspect').first()['target_aspect']
                                disclosure = MetricSchema.objects.filter(
                                    metric__icontains=target_metric).values('disclosure').first()['disclosure']
                                target_unit = MetricSchema.objects.filter(
                                    metric__icontains=target_metric).values('unit').first()['unit']
                                compulsory = MetricSchema.objects.filter(
                                    metric__icontains=target_metric).values('compulsory').first()['compulsory']
                                intensity_group = MetricSchema.objects.filter(
                                    metric__icontains=target_metric).values('intensity_group').first()[
                                    'intensity_group']
                                multiplier, converted_unit = convert_arbitrary_uom(unit, target_metric, metric_year)
                                if multiplier:
                                    try:
                                        converted_value = float(
                                            value) * float(multiplier)
                                    except:
                                        pass

                            metric_char_position = word_pos(sentence, metric)
                            number_char_position = word_pos(sentence, number)
                            if mer_count == 1:
                                split_sent = [sentence]
                            if metric in searched and number in searched:  # all entities are in the searched entities
                                continue
                            # current metric entity in the searched entities, i.e. one metric to many number case
                            elif metric in searched and number not in searched:
                                ent = [number]
                                entities.append(
                                    {'text': number, 'type': tail_entity_type,
                                     'char_position': number_char_position[0]})
                                searched.append(number)
                            # current number entity in the searched entities, i.e. one number to many metric case
                            elif number in searched and metric not in searched:
                                ent = [metric]
                                entities.append(
                                    {'text': metric, 'type': head_entity_type,
                                     'char_position': metric_char_position[0]})
                                searched.append(metric)
                            else:
                                if not metric_char_position or not number_char_position:
                                    continue
                                if metric_char_position[0] < number_char_position[0]:
                                    ent = [metric, number]
                                    entities.append(
                                        {'text': metric, 'type': head_entity_type,
                                         'char_position': metric_char_position[0]})
                                    entities.append(
                                        {'text': number, 'type': tail_entity_type,
                                         'char_position': number_char_position[0]})
                                    searched.append(metric)
                                    searched.append(number)
                                else:
                                    ent = [number, metric]
                                    entities.append(
                                        {'text': number, 'type': tail_entity_type,
                                         'char_position': number_char_position[0]})
                                    entities.append(
                                        {'text': metric, 'type': head_entity_type,
                                         'char_position': metric_char_position[0]})
                                    searched.append(metric)
                                    searched.append(number)
                            tmp_split_sent = split_sent.copy()
                            split_sent = split_sentence(split_sent, ent)
                            if split_sent == tmp_split_sent:
                                continue
                            update_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
                            if "date" not in flat_list or model_version == 'v1':  # if "date" not found in relations
                                metric_year = year
                                dic = {
                                    'metric_year': metric_year,
                                    'metric': metric,
                                    'metric_char_position': metric_char_position,
                                    'subject': subject,
                                    'target_aspect': target_aspect,
                                    'disclosure': disclosure,
                                    'target_metric': target_metric,
                                    'compulsory': compulsory,
                                    'intensity_group': intensity_group,
                                    'similarity': sim_score,
                                    'relation': relation,
                                    'number': number,
                                    'number_char_position': number_char_position,
                                    'original_value': value,
                                    'unit': unit,
                                    'uom_conversion_multiplier': multiplier,
                                    'converted_value': converted_value,
                                    'converted_unit': converted_unit,
                                    'target_unit': target_unit,
                                    'update_datetime': update_datetime
                                }
                                metric_dic.append({**sentences[i], **dic})
                                dic['update_datetime'] = update_datetime.strftime(
                                    "%d/%m/%Y, %H:%M:%S")
                                ners.append(dic)
                                sent = {'sent_id': sentences[i]['sent_id'], 'sentence': sentences[i]['sentence'],
                                        'ners': ners}
                            else:
                                dic = {
                                    'metric_year': None,
                                    'metric': metric,
                                    'metric_char_position': metric_char_position,
                                    'subject': subject,
                                    'target_aspect': target_aspect,
                                    'disclosure': disclosure,
                                    'target_metric': target_metric,
                                    'compulsory': compulsory,
                                    'intensity_group': intensity_group,
                                    'similarity': sim_score,
                                    'relation': relation,
                                    'number': number,
                                    'number_char_position': number_char_position,
                                    'original_value': value,
                                    'unit': unit,
                                    'uom_conversion_multiplier': multiplier,
                                    'converted_value': converted_value,
                                    'converted_unit': converted_unit,
                                    'target_unit': target_unit,
                                    'update_datetime': update_datetime
                                }
                    elif head_entity_type == "number":
                        number = head_entity
                        metric_year = tail_entity
                        if 'dic' in locals() and dic['number'] == number:
                            # print(
                            #     'Match number-year pair with previous metric-number pair')
                            dic['metric_year'] = metric_year
                            # print('Metric entity relation extracted: \n', json.dumps(
                            #     dic, indent=4, sort_keys=True, default=str), '\n')
                            metric_dic.append({**sentences[i], **dic})
                            dic['update_datetime'] = update_datetime.strftime(
                                "%d/%m/%Y, %H:%M:%S")
                            ners.append(dic)
                            sent = {
                                'sent_id': sentences[i]['sent_id'], 'sentence': sentences[i]['sentence'], 'ners': ners}
                if split_sent:
                    sent['split_sentence'] = []
                    prev_char_pos = [0, 0]
                    while len(entities) > 0 and len(split_sent) > 0:
                        entity = entities.pop(0)
                        frag = split_sent.pop(0)
                        if frag.strip():
                            char_pos = word_pos(sentence, frag)
                            for pos in char_pos:
                                if pos[0] < prev_char_pos[0]:
                                    continue
                                else:
                                    char_pos = pos
                            sent['split_sentence'].append(
                                {'text': frag, 'type': 'normal', 'char_position': char_pos})
                        sent['split_sentence'].append(entity)
                        prev_char_pos = entity['char_position']
                    if split_sent:
                        char_pos = word_pos(sentence, split_sent[0])
                        for pos in char_pos:
                            if pos[0] < prev_char_pos[0]:
                                continue
                            else:
                                char_pos = pos
                        sent['split_sentence'].append(
                            {'text': split_sent[0], 'type': 'normal', 'char_position': char_pos})

                if sent:
                    if 'mers' not in locals():
                        mers = {'page_id': page_id,
                                'text_block_id': text_block_id, block_element: [sent]}
                    if same_element and same_block and same_page and block_element in mers:
                        mers[block_element].append(sent)
                    elif (not same_element and same_block and same_page) or (
                            same_element and same_block and same_page and block_element not in mers):
                        mers.update({block_element: [sent]})
                    else:
                        if total_mer_count != 1:
                            metric_nested_dic['metric_entity_relations'].append(mers)
                        mers = {'page_id': page_id,
                                'text_block_id': text_block_id, block_element: [sent]}

                    prev_page_id = page_id
                    prev_text_block_id = text_block_id
                    prev_block_element = block_element

            # elif metric_entity:
            #     for j, ent in enumerate(metric_entity):
            #         entity_type, entity = ent
            #         target_metric, sim_score = get_similarity_sentbert(entity, metric_list)
            #         ent_dic = {
            #             'entity_type': entity_type,
            #             'entity': entity,
            #             'target_metric': target_metric,
            #             'similarity': sim_score
            #         }
            #         ent_data.append({**sentences[i], **ent_dic})
            #         # print('Metric entity extracted only: ', ent_dic, '\n')
            # else:
            #     ent_dic = {
            #         'entity_type': None,
            #         'entity': None,
            #         'target_metric': None,
            #         'similarity': None,
            #     }
            #     ent_data.append({**sentences[i], **ent_dic})
            # print('No metric entity found: ', ent_dic, '\n')

            if i == len(all_metrics_predictions) - 1:
                if 'mers' not in locals() and sent:
                    mers = {'page_id': page_id,
                            'text_block_id': text_block_id, block_element: [sent]}
                if 'mers' in locals():
                    metric_nested_dic['metric_entity_relations'].append(mers)
        all_end1 = timeit.default_timer()
        total_metric_extract_time = all_end1 - all_start1
        print(f'metric extraction completed ...')
        return metric_nested_dic, metric_dic

    def text_reasoning_extract():
        global reasoning_extract_time, total_reasoning_extract_time, rer_sent_count, total_rer_count, reasoning_model
        # '''Batch reasoning entity relation extraction'''
        all_reasoning_predictions = []
        all_start2 = timeit.default_timer()
        t = tqdm.tqdm(range(batch_num), desc='Text Reasoning Extraction')
        reasoning_model = init_UIE_model("reasoning_model_v1")
        for batch_counter2 in t:
            record_progress_update(t, PDFFILES_TABLENAME, filename_dict, 'process2_progress',
                                   'Text reasoning extraction is processing ... ', 0.06)
            start2 = timeit.default_timer()
            batch2 = [i['sentence'] for i in
                      sentences[batch_counter2 * UIE_BATCH_SIZE:(batch_counter2 + 1) * UIE_BATCH_SIZE]]
            reasoning_predictions = reasoning_predictor_predict(batch2)
            # response2 = requests.post(url_reasoning, json={
            #     'input': [i['sentence'] for i in sentences[batch_counter2 * UIE_BATCH_SIZE:(batch_counter2 + 1) * UIE_BATCH_SIZE]]})
            # reasoning_predictions = response2.json()

            stop2 = timeit.default_timer()
            reasoning_extract_time += stop2 - start2

            all_reasoning_predictions.extend(reasoning_predictions)
        del reasoning_model
        gc.collect()

        prev_page_id = prev_text_block_id = prev_block_element = None
        reasoning_dic = []
        ent_data2 = []
        result_targets = []
        reasoning_nested_dic = {'filename': filename, 'document_id': document_id, 'company_name': company_name,
                                'year': year,
                                'reasoning_entity_relations': []}

        for i, res2 in enumerate(all_reasoning_predictions):
            if isinstance(res2, dict) and 'relation' in res2:
                if isinstance(res2['relation'], dict) and 'string' in res2['relation']:
                    reasoning_entity_relations = res2['relation']['string']
                    for j, ent_rel in enumerate(reasoning_entity_relations):
                        relation, head_entity_type, head_entity, tail_entity_type, tail_entity = ent_rel
                        is_valid_pair = uie_reasoning_validator(head_entity_type, head_entity, tail_entity_type, tail_entity, relation)
                        if is_valid_pair:
                            result_targets.append(head_entity)

        # check if there is any valid result targets. If not, the target_kw_dict is empty
        if result_targets:
            target_keywords_count = sum([len(i) for i in metric_list])
            start_map2 = timeit.default_timer()
            target_kw_dict = get_similarity_sentbert(result_targets, metric_list, return_single_response=False)
            stop_map2 = timeit.default_timer()
            total_time_map2 = stop_map2 - start_map2
            row = {'task': 'Reasoning Target Mapping', 'filename': filename, 'num_result': len(result_targets), 'runtime': total_time_map2}
            log_task2csv('data/log/log_target_metric_mapping.csv', row)
        else:
            target_kw_dict = {}

        for i, res2 in enumerate(all_reasoning_predictions):
            if isinstance(res2, dict) and 'relation' in res2:
                if isinstance(res2['relation'], dict) and 'string' in res2['relation']:
                    reasoning_entity_relations = res2['relation']['string']
                else:
                    # print(f'Non-dictionary Response with res2["relation"]={res2["relation"]} and dtype={type(res2["relation"])}')
                    continue
            else:
                # print(f'Non-dictionary Response with res2={res2} and dtype={type(res2)}')
                continue

            reasoning_entity = res2['entity']['string']
            page_id = sentences[i]['page_id']
            text_block_id = sentences[i]['text_block_id']
            block_element = sentences[i]['block_element']
            sentence = sentences[i]['sentence']
            sent = {}

            # print(f'Processing reasoning on sentence no. {i}:\n {sentence}', '\n')
            # print(f'↖ With page_id = {page_id}, text_block_id = {text_block_id}, block_element = {block_element} \n')

            if reasoning_entity_relations:
                # print('reasoning entity relation found: \n', json.dumps(reasoning_entity_relations, indent=4), '\n')
                ners = []
                rer_sent_count += 1
                rer_count = 0
                entities = []
                split_sent = []
                searched = []
                same_element = block_element == prev_block_element
                same_block = text_block_id == prev_text_block_id
                same_page = page_id == prev_page_id

                for j, ent_rel in enumerate(reasoning_entity_relations):
                    relation, head_entity_type, head_entity, tail_entity_type, tail_entity = ent_rel
                    is_valid_pair = uie_reasoning_validator(head_entity_type, head_entity, tail_entity_type, tail_entity, relation)
                    if is_valid_pair:
                        # print(f'Relation = {relation}, Head = {head_entity}, Tail = {tail_entity} \n')
                        total_rer_count += 1
                        rer_count += 1
                        head_ent_char_pos = word_pos(sentence, head_entity)
                        tail_ent_char_pos = word_pos(sentence, tail_entity)
                        if not head_ent_char_pos or not tail_ent_char_pos:
                            continue
                        if rer_count == 1:
                            split_sent = [sentence]
                        if head_entity in searched and tail_entity in searched:  # all entities are in the searched entities
                            continue
                        elif head_entity in searched and tail_entity not in searched:  # current head entity is in the searched entities, i.e. one head to many tails case
                            ent = [tail_entity]
                            entities.append(
                                {'text': tail_entity, 'type': tail_entity_type, 'char_position': tail_ent_char_pos[0]})
                            searched.append(tail_entity)
                        elif tail_entity in searched and head_entity not in searched:  # current head entity is in the searched entities, i.e. one head to many tails case
                            ent = [head_entity]
                            entities.append(
                                {'text': head_entity, 'type': head_entity_type, 'char_position': head_ent_char_pos[0]})
                            searched.append(head_entity)
                        else:
                            if head_ent_char_pos[0] < tail_ent_char_pos[0]:
                                ent = [head_entity, tail_entity]
                                entities.append(
                                    {'text': head_entity, 'type': head_entity_type,
                                     'char_position': head_ent_char_pos[0]})
                                entities.append(
                                    {'text': tail_entity, 'type': tail_entity_type,
                                     'char_position': tail_ent_char_pos[0]})
                                searched.append(head_entity)
                                searched.append(tail_entity)
                            else:
                                ent = [tail_entity, head_entity]
                                entities.append(
                                    {'text': tail_entity, 'type': tail_entity_type,
                                     'char_position': tail_ent_char_pos[0]})
                                entities.append(
                                    {'text': head_entity, 'type': head_entity_type,
                                     'char_position': head_ent_char_pos[0]})
                                searched.append(head_entity)
                                searched.append(tail_entity)
                        tmp_split_sent = split_sent.copy()
                        split_sent = split_sentence(split_sent, ent)
                        if split_sent == tmp_split_sent:
                            continue
                        subject = target_aspect = disclosure = relavent_keyword = sim_score = None
                        if head_entity_type == 'target':
                            if head_entity in target_kw_dict:
                                relavent_keyword = target_kw_dict[head_entity]['similar_metrics']
                                sim_score = target_kw_dict[head_entity]['score']
                            else:
                                sim_score = 0
                            # relavent_keyword, sim_score = get_similarity_sentbert(head_entity, metric_list, return_single_response=False)
                            if relavent_keyword:
                                if isinstance(relavent_keyword, str):
                                    for k, v in disclosure_kw_dict.items():
                                        if relavent_keyword in v:
                                            disclosure = k
                                            target_aspect = disclosure_tgt_aspect[k]
                                            subject = disclosure_subject[k]
                                            break
                                elif isinstance(relavent_keyword, list):
                                    if len(relavent_keyword) > 3:
                                        relavent_keyword = relavent_keyword[:3]
                                        sim_score = sim_score[:3]
                                    disclosure = []
                                    target_aspect = []
                                    subject = []
                                    for k, v in disclosure_kw_dict.items():
                                        for kw in relavent_keyword:
                                            if kw in v:
                                                disclosure.append(k)
                                                target_aspect.append(disclosure_tgt_aspect[k])
                                                subject.append(disclosure_subject[k])
                                                break
                        update_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
                        dic2 = {
                            'head_entity_type': head_entity_type,
                            'head_entity': head_entity,
                            'head_entity_char_position': head_ent_char_pos,
                            'subject': subject,
                            'target_aspect': target_aspect,
                            'disclosure': disclosure,
                            'target_metric': relavent_keyword,
                            'similarity': sim_score,
                            'relation': relation,
                            'tail_entity_type': tail_entity_type,
                            'tail_entity': tail_entity,
                            'tail_entity_char_position': tail_ent_char_pos,
                            'update_datetime': update_datetime
                        }
                        # print('Reasoning entity relation extracted: \n', json.dumps(dic2, indent=4, sort_keys=True, default=str), '\n')
                        reasoning_dic.append({**sentences[i], **dic2})
                        dic2['update_datetime'] = update_datetime.strftime("%d/%m/%Y, %H:%M:%S")
                        ners.append(dic2)
                        sent = {'sent_id': sentences[i]['sent_id'], 'sentence': sentence, 'ners': ners}

                if split_sent:
                    sent['split_sentence'] = []
                    prev_char_pos = [0, 0]
                    while len(entities) > 0 and len(split_sent) > 0:
                        entity = entities.pop(0)
                        frag = split_sent.pop(0)
                        if frag.strip():
                            char_pos = word_pos(sentence, frag)
                            for pos in char_pos:
                                if pos[0] < prev_char_pos[0]:
                                    continue
                                else:
                                    char_pos = pos
                            sent['split_sentence'].append({'text': frag, 'type': 'normal', 'char_position': char_pos})
                        sent['split_sentence'].append(entity)
                        prev_char_pos = entity['char_position']
                    if split_sent:
                        char_pos = word_pos(sentence, split_sent[0])
                        for pos in char_pos:
                            if pos[0] < prev_char_pos[0]:
                                continue
                            else:
                                char_pos = pos
                        sent['split_sentence'].append(
                            {'text': split_sent[0], 'type': 'normal', 'char_position': char_pos})
                if sent:
                    if 'rers' not in locals():
                        rers = {'page_id': page_id, 'text_block_id': text_block_id, block_element: [sent]}
                    if same_element and same_block and same_page and block_element in rers:
                        rers[block_element].append(sent)
                    elif (not same_element and same_block and same_page) or (
                            same_element and same_block and same_page and block_element not in rers):
                        rers.update({block_element: [sent]})
                    else:
                        if rer_sent_count != 1:
                            reasoning_nested_dic['reasoning_entity_relations'].append(rers)
                        rers = {'page_id': page_id, 'text_block_id': text_block_id, block_element: [sent]}

                prev_page_id = page_id
                prev_text_block_id = text_block_id
                prev_block_element = block_element

            # elif reasoning_entity:
            #     disclosure = target_aspect = relavent_keyword = sim_score = None
            #     for j, ent in enumerate(reasoning_entity):
            #         entity_type, entity = ent
            #         if entity_type == 'target':
            #             if entity in target_kw_dict:
            #                 relavent_keyword = target_kw_dict[entity]['similar_metrics']
            #                 sim_score = target_kw_dict[entity]['score']
            #             else:
            #                 sim_score = 0
            #             relavent_keyword, sim_score = get_similarity_sentbert(entity, metric_list)
            #             if relavent_keyword:
            #                 for k, v in disclosure_kw_dict.items():
            #                     if relavent_keyword in v:
            #                         disclosure = k
            #                         target_aspect = disclosure_tgt_aspect[k]
            #                         break
            #         ent_dic2 = {
            #             'entity_type': entity_type,
            #             'entity': entity,
            #             'disclosure' : disclosure,
            #             'target_aspect' : target_aspect,
            #             'similarity' : sim_score
            #         }
            #         ent_data2.append({**sentences[i], **ent_dic2})
            #         print('Reasoning entity extracted only: ', ent_dic2, '\n')
            # else:
            #     ent_dic2 = {
            #         'entity_type': None,
            #         'entity': None,
            #         'disclosure': None,
            #         'target_aspect' : None,
            #         'similarity' : None
            #     }
            #     ent_data2.append({**sentences[i], **ent_dic2})
            #     print('No reasoning entity found: ', ent_dic2, '\n')

            if i == len(all_reasoning_predictions) - 1:
                if 'rers' not in locals() and sent:
                    rers = {'page_id': page_id, 'text_block_id': text_block_id, block_element: [sent]}
                if 'rers' in locals():
                    reasoning_nested_dic['reasoning_entity_relations'].append(rers)
        all_end2 = timeit.default_timer()
        total_reasoning_extract_time = all_end2 - all_start2
        print(f'reasoning extraction completed ...')
        return reasoning_nested_dic, reasoning_dic

    if do_metric_extract:
        metric_nested_dic, metric_dic = text_metric_extract()
        row = {'task': 'Metric Extraction', 'filename': filename, 'num_sentence': search_count,'num_valid_ner': total_mer_count, 'runtime': metric_extract_time, 'total_processing_time': total_metric_extract_time}
        log_task2csv('data/log/log_entity_relation_extraction.csv', row)
        table_name = METRIC_EXTRACTION_TABLE_NAME
        for item in metric_dic:
            filter_dict = {
                'company_name': item['company_name'],
                'year': item['year'],
                'page_id': item['page_id'],
                'text_block_id': item['text_block_id'],
                'block_element': item['block_element'],
                'sent_id': item['sent_id'],
                'sentence': item['sentence']
            }
            du.delete_data(table_name, filter_dict)
        du.insert_data(table_name=table_name, data_list=metric_dic)
        # print(
        #     f"Saved entity-relation extraction results of {filename} to database table {table_name}")
    else:
        outpath = os.path.join(
            METRIC_OUTPUT_JSON_DIR,
            re.sub(".pdf", ".json", filename)
        )
        with open(outpath, "r") as f:
            metric_nested_dic = json.load(f)
    if do_reasoning_extract:
        reasoning_nested_dic, reasoning_dic = text_reasoning_extract()
        row = {'task': 'Reasoning Extraction', 'filename': filename, 'num_sentence': rer_sent_count,'num_valid_ner': total_rer_count, 'runtime': reasoning_extract_time, 'total_processing_time': total_reasoning_extract_time}
        log_task2csv('data/log/log_entity_relation_extraction.csv', row)
        table_name = REASONING_EXTRACTION_TABLE_NAME
        for item in reasoning_dic:
            filter_dict = {
                'company_name': item['company_name'],
                'year': item['year'],
                'page_id': item['page_id'],
                'text_block_id': item['text_block_id'],
                'block_element': item['block_element'],
                'sent_id': item['sent_id'],
                'sentence': item['sentence']
            }
            du.delete_data(table_name, filter_dict)
        du.insert_data(table_name=table_name, data_list=reasoning_dic)
        # print(
        #     f"Saved entity-relation extraction results of {filename} to database table {table_name}")
    else:
        outpath = os.path.join(
            REASONING_OUTPUT_JSON_DIR,
            re.sub(".pdf", ".json", filename)
        )
        with open(outpath, "r") as f:
            reasoning_nested_dic = json.load(f)
    print('entity relation extraction completed ...')

    return metric_nested_dic, reasoning_nested_dic


def record_status_update(tablename, filter_dict, update_dict):
    du = DbUtil()
    du.update_data(
        tablename,
        filter_dict=filter_dict,
        update_dict=update_dict
    )


def record_progress_update(tqdm_obj, tablename, filter_dict, progress_name, message, proportion):
    '''
    update the progress of model inference by batch count
    @param tqdm_obj: tqdm decorator object
    @type tqdm_obj: class tqdm.tqdm
    @param tablename: database table name for the record
    @type tablename: str
    @param filter_dict: dictionary of filename key "filename" value "XXX.pdf"
    @type filter_dict: dict
    @param progress_name: field name of the progress in db table "tablename"
    @type progress_name: str
    @param message: message to display in the update status
    @type message: str
    @param proportion: proportion of the subprocess that contribute in the whole process
    @type proportion: float
    '''
    import re
    import datetime
    import pytz

    process_id = re.search(r'\d+', progress_name).group()
    du = DbUtil()
    all_process_progress = du.select_table(table_name=tablename, field_list=[progress_name], filter_dict=filter_dict)
    if all_process_progress is not None:
        process_progress = all_process_progress[0][progress_name]
        if process_progress is None:
            process_progress = 0
    else:
        filename = filter_dict["filename"]

        if filename[0].isdigit():
            stock_id, company_name, _ = filename.split('_')
            stock_id = int(stock_id)
            year = min([int(i) for i in re.findall('\D(20\d{2})\D?', filename, re.I)])
        else:
            if len(filename.split('_')) == 2:
                company_name, _ = filename.split('_')
                stock_id = None
            elif len(filename.split('_')) == 3 and filename.split('_')[1][0].isdigit():
                company_name, stock_id, _ = filename.split('_')
                stock_id = int(stock_id)
            elif len(filename.split('_')) == 3 and not filename.split('_')[1][0].isdigit():
                company_name, _, _ = filename.split('_')
                stock_id = None
            else:
                company_name = None
                stock_id = None
            try:
                year = min([int(i) for i in re.findall('\d+', filename, re.I) if re.match('\d{4}$', i)])
            except:
                year = None

        if re.search(r'.*Annual Report.*', filename, re.I):
            report_type = 'annualReport'
        elif re.search(r'.*Carbon Neutrality.*', filename, re.I):
            report_type = 'carbonNeutralityReport'
        else:
            report_type = 'esgReport'

        report_language = 'eng'
        try:
            industry = du.select_table(table_name='company_industry', field_list=['industry'],
                                       filter_dict={'company_name_en': company_name})[0]['industry']
        except:
            industry = None
        document_id = company_name + '_' + str(year) + '_' + report_language

        uploaded_date = datetime.datetime.today().replace(tzinfo=pytz.utc)
        status = 'Processing'
        result = [{
            'document_id': document_id,
            'filename': filename,
            'industry': industry,
            'company': company_name,
            'year': year,
            'report_language': report_language,
            'report_type': report_type,
            'uploaded_date': uploaded_date,
            'process1_status': 'File uploaded',
            'process2_status': 'File uploaded',
            'last_process_date': uploaded_date,
            'status': status
        }]
        # save basic information first
        delete_dict = {
            "document_id": document_id
        }
        try:
            du.delete_data(PDFFILES_TABLENAME, delete_dict)
        except:
            pass
        du.insert_data(
            table_name=PDFFILES_TABLENAME,
            data_list=result
        )
        process_progress = 0

    update_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
    batch_id = tqdm_obj.format_dict['n'] + 1
    batch_total = tqdm_obj.format_dict['total']
    update_dict = {
        f'process{process_id}_progress': process_progress + (1 / batch_total) * proportion,
        f'last_process{process_id}_elapsed_time': tqdm_obj.format_dict['elapsed'],
        f'process{process_id}_update_date': update_datetime
    }
    if batch_id != batch_total:
        update_dict[f'process{process_id}_status'] = f'{message} ({str(batch_id)}/{str(batch_total)})'
    else:
        process_name = re.match('(.*) is processing ... ', message).groups()[0]
        update_dict[f'process{process_id}_status'] = f'{process_name} completed'

    record_status_update(tablename, filter_dict, update_dict)


def generate_report_info(filename):
    '''
    Generate report info based on filename and filepath.
    If the file is PDF, page count is provided.
    @param filename: filename with extension of the file that store in 'data/pdf'
    @type filename: str
    '''
    filepath = get_filepath_with_filename(PDF_DIR, filename)
    fname = os.path.splitext(filename)[0]

    pdf_exist = os.path.exists(filepath)
    docparse_exist = os.path.exists(os.path.join(DOCPARSE_OUTPUT_JSON_DIR, fname + '.json'))
    table_extract_exist = os.path.exists(os.path.join(TABLE_OUTPUT_JSON_DIR, fname + '.json'))
    textMetric_exist = os.path.exists(os.path.join(METRIC_OUTPUT_JSON_DIR, fname + '.json'))
    textReasoning_exist = os.path.exists(os.path.join(REASONING_OUTPUT_JSON_DIR, fname + '.json'))

    filesize_MB = os.path.getsize(filepath) / (1024 * 1024)
    if filename.endswith('.pdf') and pdf_exist:
        doc = fitz.open(filepath)
        page_count = doc.page_count
    else:
        page_count = None

    if filename[0].isdigit():
        stock_id, company_name, _ = filename.split('_')
        stock_id = int(stock_id)
        year = min([int(i) for i in re.findall('\D(20\d{2})\D?', filename, re.I)])
    else:
        if len(filename.split('_')) == 2:
            company_name, _ = filename.split('_')
            stock_id = None
        elif len(filename.split('_')) == 3 and filename.split('_')[1][0].isdigit():
            company_name, stock_id, _ = filename.split('_')
            stock_id = int(stock_id)
        elif len(filename.split('_')) == 3 and not filename.split('_')[1][0].isdigit():
            company_name, _, _ = filename.split('_')
            stock_id = None
        else:
            company_name = None
            stock_id = None
        try:
            year = min([int(i) for i in re.findall('\d+', filename, re.I) if re.match('\d{4}$', i)])
        except:
            year = None

    if re.search(r'.*Annual Report.*', filename, re.I):
        report_type = 'annualReport'
    elif re.search(r'.*Carbon Neutrality.*', filename, re.I):
        report_type = 'carbonNeutralityReport'
    else:
        report_type = 'esgReport'

    report_lang = 'eng'

    info = {
        'filename': filename,
        'document_id': str(company_name) + '_' + str(year) + '_' + report_lang,
        'stock_id': stock_id,
        'company_name': company_name,
        'report_year': year,
        'report_type': report_type,
        'report_language': report_lang,
        'page_count': page_count,
        'filesize_mb': filesize_MB,
        'exist_pdf': pdf_exist,
        'exist_docParse': docparse_exist,
        'exist_textMetric': textMetric_exist,
        'exist_textReasoning': textReasoning_exist,
        'exist_tableMetric': table_extract_exist
    }

    return info


def pdf_analyze(filename, document_id):
    import threading

    print(f'processing: {filename}')
    filter_dict = {
        'document_id': document_id,
        'filename': filename
    }

    if TEST_INFO.get('filename'):
        page_numbers = TEST_INFO['filename']
    else:
        doc = fitz.open(get_filepath_with_filename(PDF_DIR, filename))
        page_count = doc.page_count
        print(filename, document_id)
        print('total pages: ', page_count)
        page_numbers = [i + 1 for i in range(page_count)]
    print('begin to check json file')
    # define the outpath for each task results
    table_outpath = os.path.join(
        TABLE_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename)
    )
    parsed_doc_path = os.path.join(
        DOCPARSE_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename)
    )
    text_metric_outpath = os.path.join(
        METRIC_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename))
    text_reasoning_outpath = os.path.join(
        REASONING_OUTPUT_JSON_DIR,
        re.sub(".pdf", ".json", filename))

    # check existence of results
    table_exist = os.path.exists(table_outpath)
    parsed_doc_exist = os.path.exists(parsed_doc_path)
    text_metric_exist = os.path.exists(text_metric_outpath)
    text_reasoning_exist = os.path.exists(text_reasoning_outpath)

    return_results = {'table_extract': None, 'metric_uie': None, 'reasoning_uie': None}

    def table_extract(return_results):
        start = timeit.default_timer()
        if not table_exist:
            print('table metrics are generating: ')
            try:
                update_dict = {
                    'process1_status': 'Table extraction is processing ...',
                    'process1_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
                record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
                raw_key_metrics_list = main_func(filename, page_numbers, document_id)
                with open(table_outpath, 'w') as f:
                    json.dump(raw_key_metrics_list, f, indent=4, ensure_ascii=False)
                print('table metrics are saved to: ', f)
            except Exception as e:
                traceback.print_exc()
                status_info = 'Encountered system error with following info: ' + str(e)
                status_info = (status_info[:253] + '..') if len(status_info) > 256 else status_info
                update_dict = {
                    'process1_status': status_info,
                    'process1_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process1_elapsed_time': None,
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
                record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
                return
        else:
            with open(table_outpath) as f:
                raw_key_metrics_list = json.load(f)

        key_metrics_list = []
        for m in raw_key_metrics_list:
            if m.get('similar_score'):
                if m['similar_score'] >= 0.6:
                    key_metrics_list.append(m)
        end = timeit.default_timer()
        update_dict = {
            'process1_status': 'Table extraction completed',
            'process1_progress': 1,
            'process1_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
            'last_process1_elapsed_time': end - start,
            'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
        }
        record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
        return_results['table_extract'] = key_metrics_list

    def docparse():
        start = timeit.default_timer()
        # if the pdf is not being parsed, do document parsing
        if not parsed_doc_exist:
            try:
                update_dict = {
                    'process2_status': 'Document parsing is processing ...',
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
                record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
                document_parser(filename)
                end = timeit.default_timer()
                update_dict = {
                    'process2_status': 'Document parsing completed',
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process2_elapsed_time': end - start,
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
                record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
            except Exception as e:
                traceback.print_exc()
                status_info = 'Encountered system error with following info: ' + str(e)
                status_info = (status_info[:253] + '..') if len(status_info) > 256 else status_info
                update_dict = {
                    'process2_status': status_info,
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process2_elapsed_time': None,
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
                record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
        else:
            end = timeit.default_timer()
            update_dict = {
                'process2_status': 'Document parsing completed',
                'process2_progress': 0.89,
                'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                'last_process2_elapsed_time': end - start,
                'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
            }
            record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)

    def uie(return_results):
        # if metric extraction doesn't exists in data/text_metric_json, do metric entity relation extraction
        do_metric_extract = not text_metric_exist
        do_reasoning_extract = not text_reasoning_exist
        start = timeit.default_timer()
        if do_metric_extract or do_reasoning_extract:
            if do_metric_extract and do_reasoning_extract:
                update_dict = {
                    'process2_status': 'Text metrics & reasoning extraction is processing ...',
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
            elif do_metric_extract:
                update_dict = {
                    'process2_status': 'Text reasoning extraction completed. Text metrics extraction is processing ...',
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
            else:
                update_dict = {
                    'process2_status': 'Text metrics extraction completed. Text reasoning extraction is processing ...',
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
            record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
            print('text metrics & reasoning are generating: ')
            try:
                metric_er, reasoning_er = get_entity_relation(filename, URL_METRIC_V1, URL_REASONING, document_id,
                                                              do_metric_extract=do_metric_extract,
                                                              do_reasoning_extract=do_reasoning_extract)
            except Exception as e:
                traceback.print_exc()
                status_info = 'Encountered system error with following info: ' + str(e)
                status_info = (status_info[:253] + '..') if len(status_info) > 256 else status_info
                update_dict = {
                    'process2_status': status_info,
                    'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
                    'last_process2_elapsed_time': None,
                    'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
                }
                record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
                return
            with open(text_metric_outpath, 'w') as f:
                json.dump(metric_er, f, indent=4, ensure_ascii=False)
                print('text metrics are saved to: ', f)
            with open(text_reasoning_outpath, 'w') as f2:
                json.dump(reasoning_er, f2, indent=4, ensure_ascii=False)
                print('reasoning info are saved to: ', f2)
        else:
            with open(text_metric_outpath) as f:
                metric_er = json.load(f)
            with open(text_reasoning_outpath) as f2:
                reasoning_er = json.load(f2)
        end = timeit.default_timer()
        update_dict = {
            'process2_status': 'Text metrics & reasoning extraction completed',
            'process2_progress': 1,
            'process2_update_date': datetime.datetime.today().replace(tzinfo=pytz.utc),
            'last_process2_elapsed_time': end - start,
            'last_process_date': datetime.datetime.today().replace(tzinfo=pytz.utc)
        }
        record_status_update(PDFFILES_TABLENAME, filter_dict, update_dict)
        return_results['metric_uie'] = metric_er
        return_results['reasoning_uie'] = reasoning_er

    def docparse_uie(return_results):
        docparse()
        uie(return_results)

    def process_all(return_results):
        p1 = threading.Thread(target=table_extract, args=(return_results,))
        p2 = threading.Thread(target=docparse_uie, args=(return_results,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    process_all(return_results)
    key_metrics_list = return_results['table_extract']
    print('key_metrics_list: ', key_metrics_list)
    metric_er = return_results['metric_uie']
    reasoning_er = return_results['reasoning_uie']

    print('completed')

    return key_metrics_list, metric_er, reasoning_er


def df_to_json(df):
    """
    ONLY FOR ANTD DROPDOWN FOR METRICS
    """
    subjects = df["subject"].unique().tolist()
    json_data = []
    for subject in subjects:
        subject_data = {"value": subject,
                        "label": df.loc[df["subject"] == subject, "subject_no"].unique()[0] + ' ' + subject,
                        "children": []}
        aspects = df.loc[df["subject"] == subject, "aspect"].unique().tolist()
        for aspect in aspects:
            aspect_data = {"value": aspect,
                           "label": df.loc[(df["subject"] == subject) & (df["aspect"] == aspect), "aspect_no"].unique()[
                                        0] + ' ' + aspect,
                           "children": []}
            disclosures = df.loc[(df["subject"] == subject) & (df["aspect"] == aspect), "disclosure"].tolist()
            for i, disclosure in enumerate(disclosures):
                disclosure_data = {"value": disclosure,
                                   "label": df.loc[(df["subject"] == subject) & (
                                               df["aspect"] == aspect), "disclosure_no"].tolist()[i] + ' ' + disclosure
                                   }
                aspect_data["children"].append(disclosure_data)
            subject_data["children"].append(aspect_data)
        json_data.append(subject_data)
    return json_data


if __name__ == '__main__':
    TEST_INFO = {
        "PERSTA_Environmental,SocialandGovernanceReport2020.pdf": [21],
        # "上海石油化工股份_2020CorporateSocialResponsibilityReport.pdf": ,
        "中信資源_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf": [79, 80, 81, 82],
        "中國海洋石油_2020Environmental,SocialandGovernanceReport.pdf": [47],
        "中國石油化工股份_2020SinopecCorp.SustainabilityReport.pdf": [43, 44],
        "中國石油股份_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf": [75],
        "中國神華_2020Environmental,ResponsibilityandGovernanceReport.pdf": [95, 96, 97],
        "中煤能源_ChinaCoalEnergyCSRReport2020.pdf": [25],
        #    "中石化油服_2020Environmental,Social,andGovernance(ESG)Report.pdf": ,
        "中石化煉化工程_2020Environmental,SocialandGovernanceReport.pdf": [26],
        "中能控股_Environmental,SocialandGovernanceReport2020.pdf": [33, 34],
        "元亨燃氣_Environmental,socialandgovernancereport2020_21.pdf": [17, 18],
        "兗煤澳大利亞_ESGReport2020.pdf": [31, 32],
        "兗礦能源_SocialResponsibilityReport2020OfYanzhouCoalMiningCompanyLimited.pdf": [72, 73],
        "匯力資源_2020Environmental,SocialandGovernanceReport.pdf": [7, 8, 10],
        "南南資源_Environmental,SocialandGovernanceReport2020_21.pdf": [32],
        "安東油田服務_2020SUSTAINABILITYREPORT.pdf": [25],
        "山東墨龍_2020Environmental,SocialandGovernanceReport.pdf": [23],
        "巨濤海洋石油服務_Environmental,SocialandGovernanceReport2020.pdf": [19],
        "延長石油國際_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf": [10, 12, 15, 16],
        # "惠生工程_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf": ,
        "新海能源_Environmental,SocialandGovernanceReportforYear2020.pdf": [38],
        "易大宗_Environmental,SocialandGovernanceReport2020.pdf": [33, 34, 35, 36],
        "海隆控股_2020Environmental,SocialandGovernanceReport.pdf": [37, 38, 39, 40, 41],
        #     "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf": ,
        #     "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2021.pdf": ,
        "西伯利亞礦業_Environmental,SocialandGovernanceReport2020.pdf": [4, 5, 6],
        "西伯利亞礦業_Environmental,SocialandGovernanceReport2021.pdf": [4, 5, 6],
        "金泰能源控股_Environmental,SocialandGovernanceReport2020.pdf": [6, 7],
        "陽光油砂_2020Environmental,SocialandGovernanceReport.pdf": [8, 9, 10, 11, 13, 14, 15],
        "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf": [4, 5],
    }

    model = TableDetector(
        checkpoint_path="/home/liuqy/table_extraction/table-transformer/pubtables1m_detection_detr_r18.pth")

    for filename, page_numbers in TEST_INFO.items():
        results = main_func(filename, page_numbers, model_det=model)
