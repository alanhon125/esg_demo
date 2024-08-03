import pandas as pd
import os
import glob

files_gt = glob.glob('../../csv_gt/*_content_tokens_gt.csv')
for file_gt in files_gt:
    fname = os.path.basename(file_gt).split('.')[0].split('_content_tokens')[0]
    file_tok = f'../../csv/{fname}_content_tokens.csv'

    df_gt = pd.read_csv(file_gt).drop(columns='index')
    df_tok = pd.read_csv(file_tok).reset_index().rename({'model_pred': 'fine_tune_model_pred', 'hybrid_pred': 'fine_tune_hybrid_pred'}, axis='columns')
    df_tok2 = pd.merge(df_tok,df_gt,on=["token","page_id","rule_pred","bbox"], how='left')
    df_tok2 = df_tok2.drop_duplicates(subset=["token",'bbox',"page_id","rule_pred"]).sort_values(by=['index'])
    df_tok2.to_csv("../../csv_gt/{}_content_tokens_gt.csv".format(fname), index=False, encoding='utf-8-sig')