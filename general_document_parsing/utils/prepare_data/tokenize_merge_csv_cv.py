import pandas as pd
import os
import glob

files = glob.glob('/Users/data/Documents/GitHub/pdf_parser/csv_cv/hang_seng/*_results_old.csv')
file_new = glob.glob('/Users/data/Documents/GitHub/pdf_parser/csv_cv/*_results.csv')

for file_gt in files:
    fname = os.path.basename(file_gt).split('.')[0].split('_results_old')[0]
    file_new = f'/Users/data/Documents/GitHub/pdf_parser/csv_cv/{fname}_results.csv'
df_gt = pd.read_csv(file_gt)
df = pd.read_csv(file_new)
gt_lst = df_gt.values.tolist()

# lookup truth label that identical strings were found between ground truth csv and new rule-based grouped tokens csv, append truth label to new rule-based grouped tokens df
for index, row in df.iterrows():
    for l in gt_lst:
        page_id = l[0]
        string = l[4]
        truth = l[3]
        if row['value']==string and row['page_id']==page_id:
            df.loc[index,'truth'] = truth
            gt_lst.remove(l) # remove list from lists if identical string with same page_id found
            break

# lookup truth label that substrings were found between ground truth csv and new rule-based grouped tokens csv, append truth label to new rule-based grouped tokens df
df_nan = df[df['truth'].isna()]
for index, row in df_nan.iterrows():
    should_continue = False
    for l in gt_lst:
        page_id = l[0]
        string = l[4]
        truth = l[3]
        if (string in row['value'] or row['value'] in string) and row['page_id']==page_id: # either substring match in ground truth df or new rule-based grouped tokens df
            df.loc[index, 'truth'] = truth
            should_continue = True # continue outer loop
            break # break inner loop
    if (should_continue): continue

df_nan = df[df['truth'].isna()]
print('no. of not matches: ',len(df_nan))

df_token = df.copy()
df_token['value'] = df_token['value'].apply(lambda x: [i for i in x.split(' ') if i.strip()])
df_token = df_token.explode('value') # tokenize the string and expand df
df_token.to_csv("csv_cv/{}_gt.csv".format(fname), index=False, encoding='utf-8-sig')