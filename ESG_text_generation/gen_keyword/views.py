from django.shortcuts import render
from django.http import HttpResponse
import json
import ast
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import random

model_path = "/home/data1/public/ResearchHub/ESG_text_generation/gen_keyword/all-MiniLM-L6-v2" 
model = SentenceTransformer(model_path)

# Create your views here.
def gen_keyword(request):
    input_str = request.body.decode('utf-8')
    input_dict = ast.literal_eval(input_str)
    company_name = input_dict['company_name']
    company_name = company_name.replace("'", "''")
    year = input_dict['year']
    no_of_topics = input_dict['no_of_topics']
    
    esg_topics_f = open('gen_keyword/esg_topics.json', 'r')
    esg_topics_list = json.load(esg_topics_f)
    esg_topics_f.close()
    
    esg_sketch_f = open('gen_keyword/esg_topics_sketch.json', 'r')
    esg_sketch_dict = json.load(esg_sketch_f)
    esg_sketch_f.close()
    
    for i in range(0, len(esg_topics_list)):
        esg_topics_list[i]['score'] = 0.0
        
    with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
        sql = "select * from public.reasoning_entity_relation_unfiltered"
        sql = sql + " where company_name = '" + str(company_name) + "' and year = '" + str(year) + "'"
        print (sql)
        entity_relation_df = pd.read_sql_query(sql, conn)
             
    if entity_relation_df.shape[0] > 0:
        esg_topic_list = []
        head_entity_list = entity_relation_df['head_entity']
        head_entity_embeddings = model.encode(head_entity_list, convert_to_tensor=True)
        
        for i in range(0, len(esg_topics_list)):
            esg_topic_dict = esg_topics_list[i]
            esg_topic = esg_topic_dict['topic']
            print (esg_topic)
            esg_topic_list.append(esg_topic)
            max_score = 0
            esg_topics_head_entity_list = esg_topic_dict['head entity']
            esg_topics_head_entity_embeddings = model.encode(esg_topics_head_entity_list, convert_to_tensor=True)
            cosine_scores = util.cos_sim(head_entity_embeddings, esg_topics_head_entity_embeddings)
            for m in range(0, cosine_scores.shape[0]):
                for n in range(0, cosine_scores.shape[1]):
                    if max_score < float(cosine_scores[m][n]):
                        max_score = float(cosine_scores[m][n])
            esg_topics_list[i]['score'] = max_score

    ranked_topic_list = sorted(esg_topics_list, key=lambda d: d['score'])
    
    if no_of_topics > len(ranked_topic_list):
        no_of_topics = len(ranked_topic_list)
    
    result_list = []
    for i in range(0, no_of_topics):
        esg_topic_dict = ranked_topic_list[i]
        esg_topic = esg_topic_dict['topic']
        KPI = esg_topic_dict['KPI']
        KPI_description = esg_topic_dict['KPI description']
        score = esg_topic_dict['score']
        esg_sketch_list = esg_sketch_dict[esg_topic]
        sketch = esg_sketch_list[random.randint(0, len(esg_sketch_list)-1)]
        result_dict = {"topic":esg_topic,
                        "KPI":KPI,
                        "KPI description":KPI_description,
                        "keywords":sketch}
        result_list.append(result_dict)
        
    output_dict = {}
    output_dict["success"] = True
    output_dict["result"] = result_list
    return HttpResponse(json.dumps(output_dict))
