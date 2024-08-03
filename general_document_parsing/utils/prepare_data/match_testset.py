import pandas as pd
import os
import glob
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

files = glob.glob('../../data/csv_gt/*_content_tokens_gt.csv')
file = '../../data/csv_gt/dataset_train.docparse_json'
type = file.split('/')[3].split('_')[1].split('.')[0]
seq = ['single','multi','complex','all']
single = ['西伯利亞礦業','飛尚無煙煤','金泰能源控股']
multi = ['中煤能源','上海石油化工股份','中石化煉化工程']
complex = ['中國石油股份','兗煤澳大利亞','中國神華']

test_pages = [(json.loads(line)['image_path'].split('/')[2].split('_')[0],int(json.loads(line)['image_path'].split('_')[2])) for line in open(file,'r')]
df_single = pd.DataFrame(columns={'fname','index','token','bbox','page_id','rule_pred','fine_tune_model_pred','fine_tune_hybrid_pred','model_pred','hybrid_pred','truth'})
df_multi = pd.DataFrame(columns={'fname','index','token','bbox','page_id','rule_pred','fine_tune_model_pred','fine_tune_hybrid_pred','model_pred','hybrid_pred','truth'})
df_complex = pd.DataFrame(columns={'fname','index','token','bbox','page_id','rule_pred','fine_tune_model_pred','fine_tune_hybrid_pred','model_pred','hybrid_pred','truth'})
df_all = pd.DataFrame(columns={'fname','index','token','bbox','page_id','rule_pred','fine_tune_model_pred','fine_tune_hybrid_pred','model_pred','hybrid_pred','truth'})
single_count = len([elem for elem in test_pages if elem[0] in single])
multi_count = len([elem for elem in test_pages if elem[0] in multi])
complex_count = len([elem for elem in test_pages if elem[0] in complex])

print(f'single: {single_count}; multi: {multi_count}, complex: {complex_count}')

for file_gt in files:
    fname = os.path.basename(file_gt).split('_')[0]
    tmp_test_pages = sorted([i[1] for i in test_pages if i[0]==fname])
    df_gt = pd.read_csv(file_gt)
    df_gt = df_gt.dropna().astype({'index':'int','token':'str','bbox':'object','page_id':'int','rule_pred':'str','fine_tune_model_pred':'str','fine_tune_hybrid_pred':'str','model_pred':'str','hybrid_pred':'str','truth':'str'})
    df_gt = df_gt[df_gt['page_id'].isin(tmp_test_pages)]
    df_gt['fname'] = fname
    if fname in single:
        df_single = df_single.append(df_gt, ignore_index=True)
    elif fname in multi:
        df_multi = df_multi.append(df_gt, ignore_index=True)
    else:
        df_complex = df_complex.append(df_gt, ignore_index=True)
    df_all = df_all.append(df_gt, ignore_index=True)

label_list = ['caption', 'figure', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']

for i, df in enumerate([df_single,df_multi,df_complex,df_all]):
    df.to_csv(f"../../data/csv_gt/{type}set_{seq[i]}_gt.csv", index=False, encoding='utf-8-sig')
    y_true = df['truth'].tolist()
    y_pred = df['fine_tune_model_pred'].tolist()
    y_hybrid_pred = df['fine_tune_hybrid_pred'].tolist()
    y_rule = df['rule_pred'].tolist()

    report = classification_report(y_true,y_pred, output_dict=True)
    report_hybrid = classification_report(y_true,y_hybrid_pred, output_dict=True)
    report_rule = classification_report(y_true, y_rule, output_dict=True)

    df_report = pd.DataFrame(report).transpose()
    df_report_hybrid = pd.DataFrame(report_hybrid).transpose()
    df_report_rule = pd.DataFrame(report_rule).transpose()

    df_report.to_csv(f"../../data/report/model_{type}_{seq[i]}_report.csv", index=True, encoding='utf-8-sig')
    df_report_hybrid.to_csv(f"../../data/report/hybrid_{type}_{seq[i]}_report.csv", index=True, encoding='utf-8-sig')
    df_report_rule.to_csv(f"../../data/report/rule_{type}_{seq[i]}_report.csv", index=True, encoding='utf-8-sig')

    confusion = confusion_matrix(y_true,y_pred, labels=label_list)
    confusion_hybrid = confusion_matrix(y_true,y_hybrid_pred, labels=label_list)
    confusion_rule = confusion_matrix(y_true, y_rule, labels=label_list)

    pd.DataFrame(confusion, index=label_list, columns=label_list).to_csv(f'../../data/report/model_{type}_{seq[i]}_confusion_matrix.csv', index=True, encoding='utf-8-sig')
    pd.DataFrame(confusion_hybrid, index=label_list, columns=label_list).to_csv(f'../../data/report/hybrid_{type}_{seq[i]}_confusion_matrix.csv', index=True, encoding='utf-8-sig')
    pd.DataFrame(confusion_rule, index=label_list, columns=label_list).to_csv(
        f'../../data/report/rule_{type}_{seq[i]}_confusion_matrix.csv', index=True, encoding='utf-8-sig')
