from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import glob
import os
import numpy as np

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

# files = glob.glob('/Users/data/Library/CloudStorage/OneDrive-HongKongAppliedScienceandTechnologyResearchInstituteCompanyLimited/pdf_parser/csv_gt/*_content_tokens.csv')
# outdir = '/Users/data/Library/CloudStorage/OneDrive-HongKongAppliedScienceandTechnologyResearchInstituteCompanyLimited/pdf_parser/csv_report/'

files = glob.glob('/Users/data/Documents/GitHub/pdf_parser/csv_cv/*_gt.csv')
file = '/csv_cv/hang_seng/all.csv'
outdir = '/Users/data/Documents/GitHub/pdf_parser/csv_cv/hang_seng/'
# for file in files:
fname = os.path.basename(file).split('.')[0]
print('filename: ',fname)
df = pd.read_csv(file)
y_true = df['truth'].tolist()
# y_rule_pred = df['rule_pred'].tolist()
# y_hybrid_pred = df['hybrid_pred'].tolist()
y_pred = df['type'].tolist()

label_list = ['Address', 'Course', 'Email', 'Experience', 'Name', 'Organization', 'Other', 'Phone', 'Role', 'Short_bio', 'Skill','Time']
# label_list = get_label_list(y_true)
label_to_id = {l: i for i, l in enumerate(label_list)}
label_id = list(label_to_id.values())

# ['Address', 'Course', 'Email', 'Experience', 'Name', 'Organization', 'Other', 'Phone', 'Role', 'Short_bio', 'Skill',
#  'Time']
# ['caption', 'date', 'figure', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
report = classification_report(y_true,y_pred, output_dict=True)
confusion = confusion_matrix(y_true,y_pred, labels=label_list)
pd.DataFrame(confusion, index=label_list, columns=label_list).to_csv('{}{}_confusion_matrix.csv'.format(outdir,fname), index=True, encoding='utf-8-sig')
df = pd.DataFrame(report).transpose()
# report_rule = classification_report(y_true, y_rule_pred, output_dict=True)
# df_rule = pd.DataFrame(report_rule).transpose()
# report_hybrid = classification_report(y_true, y_hybrid_pred, output_dict=True)
# df_hybrid = pd.DataFrame(report_hybrid).transpose()

df.to_csv("{}{}{}_report.csv".format(outdir,fname,'_model'), index=True, encoding='utf-8-sig')
# df_rule.to_csv("{}{}{}_report.csv".format(outdir, fname, '_rule'), index=True, encoding='utf-8-sig')
# df_hybrid.to_csv("{}{}{}_report.csv".format(outdir, fname, '_hybrid'), index=True, encoding='utf-8-sig')
