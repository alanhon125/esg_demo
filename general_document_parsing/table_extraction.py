from api.config import *
from sentence_transformers import SentenceTransformer, util
import camelot
import re
import pandas as pd
import os
import json


# rule based table extraction

def get_parsed_doc(filename):
    outpath = os.path.join(
        DOCPARSE_OUTPUT_JSON_DIR,
        re.sub('.pdf', '.json', filename)
    )
    with open(outpath, 'r') as f:
        output = json.load(f)
    heading_regex = '^title|section(_){0,1}\d{0,}$'
    parsed_doc = []
    for item in output['content']:
        for k, v in item.items():
            if re.search(heading_regex, k):
                dic = {}
                dic['title'] = item[k]
                dic['page_range'] = item['child_page_range']
                dic['filename'] = output['filename']
                parsed_doc.append(dic)
    return parsed_doc


def find_env_related_parts(parsed_doc):
    env_pos = []
    for item in parsed_doc:
        for word in KEY_ENV_WORDS:
            if re.search(word, item['title'], re.I):
                if item not in env_pos:
                    env_pos.append(item)
    return env_pos


def save_possible_table_page(env_pos):
    excel_filenames = []
    for item in env_pos:
        try:
            page_range = item['page_range']
            if not item['filename'].endswith('pdf'):
                inpath = os.path.join(PDF_DIR, item['filename'] + '.pdf')
            else:
                inpath = os.path.join(PDF_DIR, item['filename'])
            if page_range:
                for page in range(page_range[0], page_range[1] + 1):
                    # try the default flavor=lattice method first
                    num_of_tables = 0
                    table_list = camelot.read_pdf(
                        inpath,
                        pages=str(page),
                    )
                    if table_list:
                        num_of_tables = table_list.n
                        if num_of_tables > 0:
                            for n in range(0, num_of_tables):
                                table_df = pd.DataFrame()
                                table_df = table_list[n].df
                                if len(list(table_df.columns)) > 1:
                                    f_name = os.path.join(
                                        OUTPUT_TABLE_DETECTION_DIR,
                                        '{}_{}_{}.xlsx'.format(item['filename'], page, n)
                                    )
                                    table_df.to_excel(f_name, encoding='utf8', index=False)
                                    excel_filenames.append(f_name)
                        print('lattice', item['filename'], item['title'], page, num_of_tables)
                    else:
                        table_list = camelot.read_pdf(
                            inpath,
                            pages=str(page),
                            flavor='stream',
                            row_tol=3
                        )
                        if table_list:
                            num_of_tables = table_list.n
                            if num_of_tables > 0:
                                for n in range(0, num_of_tables):
                                    table_df = pd.DataFrame()
                                    table_df = table_list[n].df
                                    if len(list(table_df.columns)) > 1:
                                        f_name = os.path.join(
                                            OUTPUT_TABLE_DETECTION_DIR,
                                            '{}_{}_{}.xlsx'.format(item['filename'], page, n)
                                        )
                                        table_df.to_excel(f_name, encoding='utf8', index=False)
                                        excel_filenames.append(f_name)
                            print('stream', item['filename'], item['title'], page, num_of_tables)
        except Exception as e:
            print(e, item['filename'])
    return list(set(excel_filenames))


def extract_table_text(excel_filenames):
    detections = []
    i = 1

    for f in excel_filenames:
        try:
            df = pd.read_excel(f)
            df = df.fillna('')
            records = df.to_dict(orient='records')
            column_length = len(df.columns)
            detection = {
                'id': i,
                'filename': f,
                'items': [item for item in records if item[column_length - 1] != '']
            }
            i += 1
            if len(detection['items']) > 0:
                detections.append(detection)
        except Exception as e:
            print('extract table text:', e)

    table_text_list = []

    for detection in detections:
        try:
            titles = []
            contents = []
            items = detection['items']
            i = 0
            for item in items:
                if item[0] == '':
                    titles.append(item)
                    i += 1
                else:
                    titles.append(item)
                    i += 1
                    break
            if not titles:
                titles = [items[0]]
                contents = items[1:]
            else:
                contents = items[i:]
            # process titles
            dict_ = {}
            if len(titles) > 1:
                for key in titles[0].keys():
                    dict_.update({key: ' '.join(title[key] for title in titles)})
                new_titles = [dict_]
            else:
                new_titles = titles

            result = []

            for k, v in contents[0].items():
                if re.search('[A-Za-z]+', v):
                    j = k
                    break
            for content in contents:
                name = content[j]
                dic = {}
                dic['name'] = name
                for k, v in content.items():
                    if k != j:
                        dic.update({
                            new_titles[0][k].lower(): v
                        })
                        if dic not in result:
                            result.append(dic)
            table_text_list.append({detection['filename']: result})
        except Exception as e:
            print(e, detection['filename'])
        # TODO: add logs
    return table_text_list


def extract_key_metrics(table_text_list):
    key_metrics_list = []
    similar_pairs = {}
    names = []
    for table_text in table_text_list:
        for k, v in table_text.items():
            names.extend([item['name'] for item in v])

    names = list(set(names))
    MODEL = SentenceTransformer("models/checkpoints/all-MiniLM-L6-v2")
    embeddings1 = MODEL.encode(list(set(names)), convert_to_tensor=True)
    embeddings2 = MODEL.encode(SCHEMA, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    for i in range(0, len(names)):
        max_j = list(cosine_scores[i]).index(max(cosine_scores[i]))
        # if max(cosine_scores[i]) > 0.35:
        similar_pairs.update({
            names[i]: {
                'similar_metrics': SCHEMA[max_j],
                'score': max(cosine_scores[i]).item()
            }
        })

    for table_text in table_text_list:
        for filename, l in table_text.items():
            for item in l:
                dic = {}
                # dic['filename'] = filename.split('/')[-1]
                dic['position'] = filename.split('/')[-1]
                dic['company_name'] = filename.split('/')[-1].split('_')[0]
                dic['metric'] = item['name']
                dic['target_metric'] = similar_pairs.get(item['name'])['similar_metrics']
                dic['similar_score'] = similar_pairs.get(item['name'])['score']
                if 'unit' in item.keys():
                    dic['unit'] = item['unit']
                for k in item.keys():
                    if re.search('\d\d\d\d', k):
                        dic['year'] = re.findall('\d\d\d\d', k, re.I)[0]
                        if len(item[k].split(' ')) > 1:
                            dic['value'] = item[k].split(' ')[0]
                            if not dic.get('unit'):
                                dic['unit'] = item[k].split(' ')[-1]
                        else:
                            dic['value'] = item[k]
                        if dic not in key_metrics_list:
                            key_metrics_list.append(dic)
    return key_metrics_list


if __name__ == '__main__':
    file_names = [
        '中信資源_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf',
        '金泰能源控股_Environmental,SocialandGovernanceReport2020.pdf',
        '匯力資源_2020Environmental,SocialandGovernanceReport.pdf',
        '巨濤海洋石油服務_Environmental,SocialandGovernanceReport2020.pdf',
        '西伯利亞礦業_Environmental,SocialandGovernanceReport2020.pdf',
        '西伯利亞礦業_Environmental,SocialandGovernanceReport2021.pdf',
        '海隆控股_2020Environmental,SocialandGovernanceReport.pdf',
        '飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf',
        '陽光油砂_2020Environmental,SocialandGovernanceReport.pdf'
    ]

    # example extractions from document parser
    filename = '飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf'
    parsed_doc = get_parsed_doc(filename)
    env_pos = find_env_related_parts(parsed_doc)
    excel_filenames = save_possible_table_page(env_pos)
    table_text_list = extract_table_text(excel_filenames)
    key_metrics_list = extract_key_metrics(table_text_list)
