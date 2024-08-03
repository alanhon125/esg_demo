from django.shortcuts import render
from django.http import HttpResponse
import json
import ast
import psycopg2
import pandas as pd
import random
import numpy as np
import pprint
import requests
import traceback

# Create your views here.

def get_first_metric(row):
    metric_list = ast.literal_eval(str(row["metric"]))
    metric = metric_list[0]
    return metric

def get_first_unit(row):
    unit = str(row["unit"])
    try:
        unit_list = ast.literal_eval(unit)
        unit = unit_list[0]
    except:
        unit = unit
    if unit == "None":
        unit = ""
    return unit

def check_metric_checklist_df(metric_checklist_df):
    compliance_check_dict = {}
    groups = metric_checklist_df.groupby("disclosure_no")
    for disclosure_no, group in groups:
        complied = False
        have_any = False
        miss_compulsory = False
        message_list = []
        related_info_list = []
        for i in range(0, group.shape[0]):
            row = group.iloc[i]
            metric = row["metric"]
            check_type = row["check_type"]
            value = row["value"]
            unit = row["unit"]
            if str(value) != "nan" :
                related_info = {"info_type":"metric",
                                "metric":metric,
                                "value":str(value) + " " + unit}
                related_info_list.append(related_info)
                if check_type in ["any", "compulsory"]:
                    have_any = True
            else:
                if check_type == "compulsory":
                    miss_compulsory = True
                    message = "Missing " + metric + " data"
                    message_list.append(message)
            if have_any:
                complied = True
            if miss_compulsory:
                complied = False
        if complied == True:
            message = "Relevant metric reported."
            message_list.append(message)
        else:
            if not miss_compulsory:
                message = "Relevant metric not found."
                message_list.append(message)
            
        compliance_check = {"KPI":disclosure_no,
                            "complied":complied,
                            "messages":message_list,
                            "related_info":related_info_list}
        compliance_check_dict[disclosure_no] = compliance_check
    return compliance_check_dict

def check_text_checklist_df(text_checklist_df):
    KPI_list_df = pd.read_csv("compliance/text_checklist.csv")
    KPI_list = list(KPI_list_df["disclosure_no"])
    compliance_check_dict = {}
    for KPI in KPI_list:
        compliance_check = {"KPI":KPI,
                            "complied":False,
                            "messages":["Related text not found"],
                            "related_info":[]}
        compliance_check_dict[KPI] = compliance_check
    groups = text_checklist_df.groupby("KPI")
    for KPI, group in groups:
        related_info_list = []
        for i in range(0, group.shape[0]):
            row = group.iloc[i]
            sentence = row["sentence"]
            metric = row["metric"]
            related_info = {"info_type":"text",
                            "metric":metric,
                            "value":sentence}
            related_info_list.append(related_info)
        compliance_check = {"KPI":KPI,
                            "complied":True,
                            "messages":["Related text found"],
                            "related_info":related_info_list}
        compliance_check_dict[KPI] = compliance_check
    return compliance_check_dict
    
def compliance(request):
    output_dict = {}
    try:
        input_str = request.body.decode('utf-8')
        input_dict = ast.literal_eval(input_str)
        company_name = input_dict["company_name"]
        # print (company_name)
        year = input_dict["year"]
    
        ### ======================= Check metric KPI ======================= 
        KPI_df = pd.read_csv("compliance/ESG_KPI.csv")
        metric_checklist_df = pd.read_csv("compliance/metric_checklist.csv")
        KPI_description_dict = {}
        for i in range(0, KPI_df.shape[0]):
            row = KPI_df.iloc[i]
            KPI = row["KPI"]
            KPI_description = row["KPI_description"]
            KPI_description_dict[KPI] = KPI_description
        
        with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
            sql = "select * from public.all_data"
            sql = sql + " where company_name = '" + company_name.replace("'","''") + "' and year = '" + str(year) + "'"
            data_df = pd.read_sql_query(sql, conn)
            
        with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
            sql = "select * from public.metric_schema"
            metric_schema_df = pd.read_sql_query(sql, conn)
            
        metric_schema_df = metric_schema_df[["metric", "unit"]]
        metric_schema_df['metric'] = metric_schema_df.apply(get_first_metric, axis=1)
        metric_schema_df['unit'] = metric_schema_df.apply(get_first_unit, axis=1)
    
        data_df = data_df.merge(metric_schema_df, on='metric', how='left')                
        metric_checklist_df = metric_checklist_df.merge(data_df, on='metric', how='left')
        metric_checklist_df = metric_checklist_df[["metric", "disclosure_no", "check_type", "value", "unit"]]    
        # metric_checklist_df.to_csv("out.csv", index = False)  
        metric_compliance_check_dict = check_metric_checklist_df(metric_checklist_df)
        
        for KPI in metric_compliance_check_dict.keys():
            compliance_check = metric_compliance_check_dict[KPI]
            KPI_description = KPI_description_dict[KPI]
            compliance_check["KPI_description"] = KPI_description
            metric_compliance_check_dict[KPI] = compliance_check
        
        ### ======================= Check text KPI =======================  
        with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
            sql = "select * from public.target_extraction"
            sql = sql + " where company_name = '" + company_name.replace("'","''") + "' and year = '" + str(year) + "'"
            data_df = pd.read_sql_query(sql, conn)
            
        text_checklist_df = pd.read_csv("compliance/text_checklist.csv")
        
        data_df = data_df[["sentence", "string_entity"]]
        
        sentence_list = []
        metric_list = []
        string_entity_list = []
        KPI_list = []
        
        for i in range(0, data_df.shape[0]):
            i_row = data_df.iloc[i]
            sentence = i_row["sentence"]
            string_entity = str(i_row["string_entity"])
            entity_dict = ast.literal_eval(string_entity)
            if "method" in entity_dict.keys():
                for j in range(0, text_checklist_df.shape[0]):
                    j_row = text_checklist_df.iloc[j]
                    entity = j_row["entity"]
                    keyword = j_row["keyword"]
                    KPI = j_row["disclosure_no"]
                    if entity in entity_dict.keys():
                        value_list = ast.literal_eval(str(entity_dict[entity]))
                        for check_value in value_list:
                            if keyword in check_value:
                                sentence_list.append(sentence)
                                metric_list.append(check_value)
                                string_entity_list.append(string_entity)
                                KPI_list.append(KPI)
        text_checklist_df = pd.DataFrame({"sentence":sentence_list,
                                          "metric":metric_list,
                                          "string_entity":string_entity_list,
                                          "KPI":KPI_list})
                    
        text_checklist_df = text_checklist_df.drop_duplicates(subset = ["sentence", "KPI"])
    
        text_compliance_check_dict = check_text_checklist_df(text_checklist_df)
    
        for KPI in text_compliance_check_dict.keys():
            compliance_check = text_compliance_check_dict[KPI]
            KPI_description = KPI_description_dict[KPI]
            compliance_check["KPI_description"] = KPI_description
            text_compliance_check_dict[KPI] = compliance_check
    
        ## ======================= Output =======================
        metric_compliance_check_dict.update(text_compliance_check_dict)
        KPI_list = list(metric_compliance_check_dict.keys())
        KPI_list.sort()
        compliance_check_list = []
        for KPI in KPI_list:
            compliance_check = metric_compliance_check_dict[KPI]
            compliance_check_list.append(compliance_check)
            
        output_dict["success"] = True  
        output_dict["compliance_check"] = compliance_check_list
        # with open("output.json", "w") as outfile:
        #     json.dump(output_dict, outfile, indent = 4)
    except Exception as e:
        print(e)
        output_dict["success"] = False
        
    return HttpResponse(json.dumps(output_dict))

def update_compliance_summary(request):
    output_dict = {}
    api_url = "http://10.6.55.126:8081/compliance/compliance/"
    # api_url = "http://127.0.0.1:8000/compliance/compliance/"
    # with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
    #     sql = "select distinct company_name, year from public.all_data order by company_name, year"
    #     all_data_df = pd.read_sql_query(sql, conn)
        
    # with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
    #     sql = "select distinct company_name, year from public.target_extraction order by company_name, year"
    #     target_extraction_df = pd.read_sql_query(sql, conn)

    with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
        sql = "select distinct company_name, year from public.compliance_detail order by company_name, year;"
        detailed_df = pd.read_sql_query(sql, conn).astype({'company_name': str, 'year': int})
    
    with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
        sql = "select distinct company_name, year from public.company_compliance_summary order by company_name, year; "
        company_compliance_summary_df = pd.read_sql_query(sql, conn).astype({'company_name': str, 'year': int})
    
    company_year_df = pd.merge(detailed_df, company_compliance_summary_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1).dropna()

    with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
        cursor = conn.cursor()
        # for i in range(0, 2):
        for i in range(0, company_year_df.shape[0]):
            print (str(i) + " out of " + str(company_year_df.shape[0]))
            try:
                row = company_year_df.iloc[i]
                company_name = row["company_name"]
                year = row["year"]
                num_of_env_kpi_complied = 0
                num_of_social_kpi_complied = 0
                num_of_aspect_A1_KPI_complied = 0
                num_of_aspect_A2_KPI_complied = 0
                num_of_aspect_A3_KPI_complied = 0
                num_of_aspect_A4_KPI_complied = 0
                num_of_aspect_B1_KPI_complied = 0
                num_of_aspect_B2_KPI_complied = 0
                num_of_aspect_B3_KPI_complied = 0
                num_of_aspect_B4_KPI_complied = 0
                num_of_aspect_B5_KPI_complied = 0
                num_of_aspect_B6_KPI_complied = 0
                num_of_aspect_B7_KPI_complied = 0
                num_of_aspect_B8_KPI_complied = 0
                input_dict = {"company_name":str(company_name), "year":int(year)}
                result = requests.post(api_url, json=input_dict).json()
                compliance_check_list = result["compliance_check"]
                
                for compliance_check in compliance_check_list:
                    KPI = compliance_check["KPI"]
                    complied = compliance_check["complied"]
                    if complied == True:
                        if KPI[0:2] == "A1":
                            num_of_env_kpi_complied += 1
                            num_of_aspect_A1_KPI_complied += 1
                        elif KPI[0:2] == "A2":
                            num_of_env_kpi_complied += 1
                            num_of_aspect_A2_KPI_complied += 1
                        elif KPI[0:2] == "A3":
                            num_of_env_kpi_complied += 1
                            num_of_aspect_A3_KPI_complied += 1
                        elif KPI[0:2] == "A4":
                            num_of_env_kpi_complied += 1
                            num_of_aspect_A4_KPI_complied += 1
                        elif KPI[0:2] == "B1":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B1_KPI_complied += 1
                        elif KPI[0:2] == "B2":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B2_KPI_complied += 1
                        elif KPI[0:2] == "B3":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B3_KPI_complied += 1
                        elif KPI[0:2] == "B4":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B4_KPI_complied += 1
                        elif KPI[0:2] == "B5":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B5_KPI_complied += 1
                        elif KPI[0:2] == "B6":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B6_KPI_complied += 1
                        elif KPI[0:2] == "B7":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B7_KPI_complied += 1
                        elif KPI[0:2] == "B8":
                            num_of_social_kpi_complied += 1
                            num_of_aspect_B8_KPI_complied += 1
                total_num_of_kpi_complied = num_of_env_kpi_complied + num_of_social_kpi_complied
                
                sql = 'INSERT INTO public.company_compliance_summary (company_name, "year", '
                sql = sql +"num_of_env_kpi_complied, "
                sql = sql + "num_of_social_kpi_complied, " 
                sql = sql + "total_num_of_kpi_complied, "
                sql = sql + "num_of_aspect_A1_KPI_complied, "
                sql = sql + "num_of_aspect_A2_KPI_complied, "
                sql = sql + "num_of_aspect_A3_KPI_complied, "
                sql = sql + "num_of_aspect_A4_KPI_complied, "
                sql = sql + "num_of_aspect_B1_KPI_complied, "
                sql = sql + "num_of_aspect_B2_KPI_complied, "
                sql = sql + "num_of_aspect_B3_KPI_complied, "
                sql = sql + "num_of_aspect_B4_KPI_complied, "
                sql = sql + "num_of_aspect_B5_KPI_complied, "
                sql = sql + "num_of_aspect_B6_KPI_complied, "
                sql = sql + "num_of_aspect_B7_KPI_complied, "
                sql = sql + "num_of_aspect_B8_KPI_complied)"
                sql = sql + " VALUES('" + company_name.replace("'","''") +"', '" + str(year) + "', "
                sql = sql + str(num_of_env_kpi_complied) + ", "
                sql = sql + str(num_of_social_kpi_complied) + ", "
                sql = sql + str(total_num_of_kpi_complied) + ", "
                sql = sql + str(num_of_aspect_A1_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_A2_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_A3_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_A4_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B1_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B2_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B3_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B4_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B5_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B6_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B7_KPI_complied) + ", "
                sql = sql + str(num_of_aspect_B8_KPI_complied) + ")"
                sql = sql + " ON CONFLICT (company_name, year)"
                sql = sql + " DO UPDATE SET num_of_env_kpi_complied = " + str (num_of_env_kpi_complied) 
                sql = sql + ", num_of_social_kpi_complied = " + str(num_of_social_kpi_complied)
                sql = sql + ", total_num_of_kpi_complied = " + str(total_num_of_kpi_complied)
                sql = sql + ", num_of_aspect_A1_KPI_complied = " + str(num_of_aspect_A1_KPI_complied)
                sql = sql + ", num_of_aspect_A2_KPI_complied = " + str(num_of_aspect_A2_KPI_complied)
                sql = sql + ", num_of_aspect_A3_KPI_complied = " + str(num_of_aspect_A3_KPI_complied)
                sql = sql + ", num_of_aspect_A4_KPI_complied = " + str(num_of_aspect_A4_KPI_complied)
                sql = sql + ", num_of_aspect_B1_KPI_complied = " + str(num_of_aspect_B1_KPI_complied)
                sql = sql + ", num_of_aspect_B2_KPI_complied = " + str(num_of_aspect_B2_KPI_complied)
                sql = sql + ", num_of_aspect_B3_KPI_complied = " + str(num_of_aspect_B3_KPI_complied)
                sql = sql + ", num_of_aspect_B4_KPI_complied = " + str(num_of_aspect_B4_KPI_complied)
                sql = sql + ", num_of_aspect_B5_KPI_complied = " + str(num_of_aspect_B5_KPI_complied)
                sql = sql + ", num_of_aspect_B6_KPI_complied = " + str(num_of_aspect_B6_KPI_complied)
                sql = sql + ", num_of_aspect_B7_KPI_complied = " + str(num_of_aspect_B7_KPI_complied)
                sql = sql + ", num_of_aspect_B8_KPI_complied = " + str(num_of_aspect_B8_KPI_complied)
                sql = sql + ";"
                
                cursor.execute(sql)
                conn.commit()
            except Exception as e:
                print (e)
                print ("Error at: " + str(i) + "/" + str(company_year_df.shape[0]) + ":[" + company_name + "," + str(year) + "]: ")
                output_dict["success"] = False
                print(str(e) + traceback.format_exc())
        cursor.close()
        output_dict["success"] = True
    return HttpResponse(json.dumps(output_dict))

def get_compliance_summary(request):
    output_dict = {}
    output_list = []
    try:
        with psycopg2.connect(user="esg_analytics", password="esg_analytics_password", host="10.6.55.126", port="5432", database="esg_analytics") as conn:
            sql = "select * from public.company_compliance_summary order by company_name, year"
            company_compliance_summary_df = pd.read_sql_query(sql, conn)
            column_list = company_compliance_summary_df.columns
            
            for i in range(0, company_compliance_summary_df.shape[0]):
                row = company_compliance_summary_df.iloc[i]
                temp_dict = {}
                for column in column_list:
                    if column in ["company_name", "year"]:
                        temp_dict[column] = str(row[column])
                    else:
                        temp_dict[column] = int(row[column])
                output_list.append(temp_dict)
        output_dict["success"] = True
    except Exception as e:
        print (e)
        output_dict["success"] = False
    output_dict["result"] = output_list
    return HttpResponse(json.dumps(output_dict))