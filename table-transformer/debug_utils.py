import json
import random
import os
import pandas as pd


def group_by_company_code(tab_metrics):
    code_metric_dict = {}
    for metric in tab_metrics:
        code = metric["company_name"]
        if code in code_metric_dict.keys():
            code_metric_dict[code].append(metric)
        else:
            code_metric_dict[code] = [metric]
    return code_metric_dict


def extract_property(tab_metrics, property_name):
    data = []
    for m in tab_metrics:
        data.append(m[property_name])
    return data


def metric_to_df(tab_metrics):
    tab_metrics.sort(key=lambda x: (int(x["company_name"]), x["page_no"]))

    company_name = extract_property(tab_metrics, "company_name")
    page_no = extract_property(tab_metrics, "page_no")
    metric = extract_property(tab_metrics, "metric")
    year = extract_property(tab_metrics, "year")
    value = extract_property(tab_metrics, "value")
    unit = extract_property(tab_metrics, "unit")
    score = extract_property(tab_metrics, "score")
    table_bbox = extract_property(tab_metrics, "enlarged_bbox")
    raw_value = extract_property(tab_metrics, "raw_value")
    raw_unit = extract_property(tab_metrics, "raw_unit")
    target_metric = extract_property(tab_metrics, "target_metric")
    similarity = extract_property(tab_metrics, "similar_score")
    label = extract_property(tab_metrics, "label")

    table_bbox = [str(t) for t in table_bbox]

    return pd.DataFrame(data={
        "company_name": company_name,
        "page_no": page_no,
        "metric": metric,
        "year": year,
        "value": value,
        "unit": unit,
        "score": score,
        "table_bbox": table_bbox,
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "target_metric": target_metric,
        "similarity": similarity,
        "label": label
    })


def metric_json_to_excel(tsr_json, output_fname):
    with open(tsr_json, "r") as f:
        tab_metrics = json.load(f)
    metric_df = metric_to_df(tab_metrics)
    metric_df.to_excel(output_fname, index=False, engine='xlsxwriter')


TSR_DEBUG_2020 = "/home/liuqy/table_extraction/tsr_debug_2020/"
TSR_DEBUG_2021 = "/home/liuqy/table_extraction/tsr_debug_2021/"


def gen_random_evaluation_set(num, curr=True):
    if curr:
        code_list = os.listdir(TSR_DEBUG_2021)
    else:
        code_list = os.listdir(TSR_DEBUG_2020)
    
    sample_size = min(num, len(code_list))
    idx = random.sample(range(len(code_list)), sample_size)
    sample_res = []
    for i in idx:
        sample_res.append(code_list[i])
    sample_res.sort()
    return sample_res


def collect_debug_figures(code_list, output_path, curr=True):
    if curr:
        debug_path = TSR_DEBUG_2021
    else:
        debug_path = TSR_DEBUG_2020
    cmd = f"tar -xzf {output_path}"
    for code in code_list:
        cmd += f" {debug_path}{code}"
    os.system(cmd)

