from core import TableDetector, TableRecognizer
from utils import table_detection, analyze_tables
import pandas as pd
import nltk


def match_exact(str1: str, str2: str):
    return str1.find(str2) != -1


def match_edit_distance(str1: str, str2: str, threshold=2):
    edit_dist = nltk.edit_distance(str1, str2)
    return edit_dist <= threshold





def load_target_metrics(path: str):
    df = pd.read_csv(path)
    target_table_list = df["table_name"].drop_duplicates().to_list()
    target_metric_list = df["metric_name"].to_list()
    return target_metric_list, target_table_list


def match_simple_table(df: pd.DataFrame, row_name: str):
    if df.shape[1] == 2:
        for idx, row in df.iterrows():
            if row[0].find(row_name) != -1:
                return row[1]
        if df.columns[0].find(row_name) != -1:
            return df.columns[1]
    return None


def match_table(df: pd.DataFrame, row_name: str, col_name: str):
    for idx, row in df.iterrows():
        if row[0].find(row_name) != -1:
            return row[col_name]
    return None


DETECTION_THRESHOLD = 0.9
RECOGNITION_THRESHOLD = 0.7


def find_all_metrics(pdf_path: str, 
                     model_det: TableDetector, 
                     model_tsr: TableRecognizer,
                     debug: bool = False):
    # table detection
    print("table detection...")
    tables, _, _ = table_detection(pdf_path, model_det, DETECTION_THRESHOLD, keywords=[])
    
    # table structure recognition
    print("table structure analysis...")
    tsr_results = analyze_tables(model_tsr, pdf_path, tables, RECOGNITION_THRESHOLD, simple=True)
    assert len(tsr_results[0]) == 2

    # match metrics
    print("match target metrics...")
    matched = {}
    for tab_info, df in tsr_results:
        if df.shape[1] == 2:
            # simple two-column table
            matched[df.columns[0]] = df.columns[1]
            for idx, row in df.iterrows():
                matched[row[0]] = row[1]
        elif df.shape[1] > 2:
            for idx, row in df.iterrows():
                for col in df.columns[1:]:
                    matched[f"{row[0]}-{col}"] = row[col]
    print(f"found {len(matched)} metrics in total")
    return matched


def match_target(pdf_path: str, 
                 target_metric_path: str, 
                 model_det: TableDetector, 
                 model_tsr: TableRecognizer,
                 debug: bool = False):
    # load target metric and corresponding table names
    target_metric_list, target_table_list = load_target_metrics(target_metric_path)
    
    # table detection
    print("table detection...")
    tables, _, _ = table_detection(pdf_path, model_det, DETECTION_THRESHOLD, keywords=target_table_list)
    
    # table structure recognition
    print("table structure analysis...")
    tsr_results = analyze_tables(model_tsr, pdf_path, tables, RECOGNITION_THRESHOLD, simple=True)
    tsr_tables = [t[1] for t in tsr_results]

    # match metrics
    print("match target metrics...")
    matched_result = {}
    unmatched_metrics = []
    for metric_name in target_metric_list:
        found_flag = False
        for df in tsr_tables:
            simple_match_res = match_simple_table(df, metric_name)
            if simple_match_res is not None:
                found_flag = True
                matched_result[metric_name] = simple_match_res
            else:
                for col_name in df.columns[1:]:
                    general_match_res = match_table(df, metric_name, col_name)
                    if general_match_res is not None:
                        found_flag = True
                        matched_result[f"{metric_name}-{col_name}"] = general_match_res
            if found_flag:
                break
        if not found_flag:
            unmatched_metrics.append(metric_name)

    print(f"#matched metrics: {len(matched_result)} #unmatched metrics: {len(unmatched_metrics)}")

    return matched_result, unmatched_metrics


def write_metrics(metric_dict: dict, output_path: str):
    metric_names = []
    metric_vals = []
    for key, val in metric_dict.items():
        if isinstance(val, str):
            metric_names.append(key)
            metric_vals.append(val)
    data = {"metric": metric_names, "value": metric_vals}
    df = pd.DataFrame(data=data)
    df.to_excel(output_path, index=False)


