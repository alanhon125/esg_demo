import json
import re
import os
import multiprocessing
import sys
import requests
import datetime
from django.shortcuts import render
from django.http import HttpResponse
from django.db.models import Q
from postgresql_storage.db_util import DbUtil
from asgiref.sync import sync_to_async
from api.config import *
from api.main import *
from general_document_parsing.info_ext_class import *
from uie_tools.postprocessing import *
import traceback
import pandas as pd
import numpy as np
import fitz
import ast
from uie_tools.utils import *
from uie_tools.uom_conversion import *
from postgresql_storage.metric_extraction_models import MetricSchema
from postgresql_storage.reasoning_extraction_models import TargetAspect
import itertools
import pytz

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

async def generate_doc_parser_result(request):
    """
    Perform or extract document parsing output with filename provided
    Prerequisite: Upload pdf file to data/pdf
    request = {"filename": FILENAME.pdf} or {"filename": [List of FILENAME.pdf]}
    """
    request = json.loads(request.body.decode("utf-8"))
    print(request)
    pdf_files = request.get("filename")
    # Set use_model = True to apply LayoutLMv3 model for document token classification
    try:
        use_model = request.get("use_model")
    except:
        use_model = USE_MODEL
    # Set do_annot = True to apply PDF annotation for reading order, text element tag, text block grouping visualization
    try:
        do_annot = request.get("do_annot")
    except:
        do_annot = DOCPARSE_PDF_ANNOTATE
    try:
        document_type = request.get("document_type")
    except:
        document_type = DOCPARSE_DOCUMENT_TYPE
    if isinstance(pdf_files, str):
        files = [pdf_files]
    else:
        files = pdf_files
    need_parsed_files = []
    key_ts = ['TS', 'term sheet']
    key_fa = ['FA', 'facility agreement', 'facilities agreement']
    # check and add list of pdf filenames that don't exist document parsing result
    for pdf_file in files:
        inpath = get_filepath_with_filename(PDF_DIR, pdf_file)
        if re.match(r'.*' + r'.*|.*'.join(key_ts), pdf_file, flags=re.IGNORECASE):
            sub_folder = 'TS/'
        elif re.match(r'.*' + r'.*|.*'.join(key_fa), pdf_file, flags=re.IGNORECASE):
            sub_folder = 'FA/'
        else:
            sub_folder = 'esgReport/'
        outpath = os.path.join(
            DOCPARSE_OUTPUT_JSON_DIR + sub_folder,
            re.sub(".pdf", ".json", pdf_file)
        )
        if not os.path.exists(outpath):
            if not os.path.exists(inpath):
                response = {
                    "success": False,
                    "error": pdf_file + ' do not exist in the PDF directory. Please upload the pdf before document parsing.'
                }
                return HttpResponse(json.dumps(response, indent=4))
            else:
                need_parsed_files.append(pdf_file)
    # perform document parsing
    try:
        document_parser(need_parsed_files, use_model=use_model, do_annot = do_annot, document_type = document_type)
        # generate the success response
        if isinstance(pdf_files, str):
            outpath = os.path.join(
                DOCPARSE_OUTPUT_JSON_DIR + sub_folder,
                re.sub(".pdf", ".json", pdf_files)
            )
            with open(outpath, "r") as f:
                output = json.load(f)
            response = {"success": True, "parsed_content": output}
        else:
            response = {"success": True, "parsed_content_list": pdf_files}
    except Exception as e:
        traceback.print_exc()
        response = {
            "success": False,
            "error": str(e)
        }

    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def metrics_predict(request):
    '''
    to predict the string, location of metrics, numbers(values with unit) and relations(equal to/less than/greater than) from a sentence.
    request = {
        "input": <A SENTENCE> OR <LIST OF SENTENCES>
    }
    '''
    request = json.loads(request.body.decode("utf-8"))
    input_sent = request.get("input")
    response = metrics_predictor_predict(input_sent)
    delete_UIE_model()
    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def reasoning_predict(request):
    '''
    to predict the head entity(target), tail entity(method/guideline) and relations(reduced_by/increased_by/comply_with) from a sentence.
    request = {
        "input": <A SENTENCE> OR <LIST OF SENTENCES>
    }
    '''
    request = json.loads(request.body.decode("utf-8"))
    input_sent = request.get("input")
    response = reasoning_predictor_predict(input_sent)
    delete_UIE_model()
    return HttpResponse(json.dumps(response, indent=4))

async def generate_text_info(request):
    """
    request = {"filename": FILENAME.pdf}
    """
    try:
        request = json.loads(request.body.decode("utf-8"))
        pdf_file = request.get("filename")
        outpath = get_filepath_with_filename(DOCPARSE_OUTPUT_JSON_DIR, re.sub(".pdf", ".json", pdf_file))
        if not os.path.exists(outpath):
            document_parser(pdf_file)

        texts = get_text_from_json(outpath)

        results = extract_text_info(texts)

        post_info = json.loads(results.to_json(
            orient="records", date_format="iso"))

        response = {"success": True,
                    "ext_info": post_info}
    except Exception as e:
        traceback.print_exc()
        response = {
            "success": False,
            "error": str(e),
            "filename": pdf_file
        }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def save_text_info(request):
    '''
    request = {"filenames": [LIST OF FILENAMES.pdf]}
    '''
    request = json.loads(request.body.decode("utf-8"))
    filenames = request["filenames"]
    for pdf_file in filenames:
        try:
            print("start", pdf_file)
            outpath = get_filepath_with_filename(DOCPARSE_OUTPUT_JSON_DIR, re.sub(".pdf", ".json", pdf_file))
            if not os.path.exists(outpath):
                document_parser(pdf_file)

            texts = get_text_from_json(outpath)
            results = extract_text_info(texts)
            post_info = json.loads(results.to_json(
                orient="records", date_format="iso"))

            year = ""
            if re.findall("\d+", pdf_file):
                year = re.findall("\d+", pdf_file)[0]

            text_info = []
            if post_info:
                for item in post_info:
                    info = item["info"]
                    for w in info:
                        dic = {
                            "company_name": pdf_file.split("/")[-1].split("_")[0],
                            "metric": w["keyword"],
                            "unit": w["measurement"],
                            # TODO: convert string number to float
                            "value": w["number"],
                            "year": year
                        }
                        text_info.append(dic)
            if text_info:
                du = DbUtil()
                du.insert_data(table_name="test_text_info",
                               data_list=text_info)
                print("Saved: ", pdf_file)
            response = {"success": True,
                        "text_info": text_info}
        except Exception as e:
            traceback.print_exc()
            response = {
                "success": False,
                "error": str(e),
                "filename": pdf_file
            }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def extract_img_info(request):
    ''' for debug purpose
    Insert table metric extraction result to database table "test_environment_info" with filename-page_id pairs provided
    Prerequisite: Uploaded report pdf to data/pdf
    request = {"test_info": {"filename.pdf":[LIST OF PAGE ID],...}}
    '''
    request = json.loads(request.body.decode('utf-8'))
    test_info = request['test_info']
    result = []
    for filename, page_numbers in test_info.items():
        try:
            img_info_list = detect_table_areas(filename, page_numbers)
            with open(f'data/tables/{filename}.json', 'w') as f:
                json.dump(img_info_list, f, indent=4, ensure_ascii=False)
            result.extend(img_info_list)
        except Exception as e:
            traceback.print_exc()
            print(filename, e)
    response = {'success': True, 'result': result}
    with open('data/tables/img_info.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return HttpResponse(json.dumps(response, indent=4))


async def extract_parsed_doc(request):
    ''' Extract or perform document parsing with giving a list of filename keys
    request = {"test_info": {"filename.pdf":[LIST OF PAGE ID],...}}
    '''
    request = json.loads(request.body.decode('utf-8'))
    test_info = request['test_info']
    result = []
    for filename, page_numbers in test_info.items():
        try:
            outpath = get_filepath_with_filename(DOCPARSE_OUTPUT_JSON_DIR, re.sub(".pdf", ".json", pdf_file))
            if not os.path.exists(outpath):
                document_parser(filename)
            parsed_doc = get_parsed_doc(filename)
            with open(DOCPARSE_OUTPUT_JSON_DIR+f'parsed_doc_{filename}.json', 'w') as f:
                json.dump(parsed_doc, f, indent=4, ensure_ascii=False)
            result.extend(parsed_doc)
        except Exception as e:
            traceback.print_exc()
            print(filename, e)
    response = {'success': True}
    with open(DOCPARSE_OUTPUT_JSON_DIR+'parsed_doc.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def scp_files(request):
    '''
    SCP files from server to local with giving filenames and source directory path
    Prerequisite: Filename exists in source directory path
    request = {
                "user": "data",
                "server": <YOUR_LOCAL_IP>,
                "password": <YOUR_LOCAL_USER_PASSWORD">,
                "port": <YOUR_PORT>,
                "server_dirpath": <YOUR_LOCAL_DESTINATION_DIRECTORY>,
                "host_dirpath": <HOST_SERVER_DESTINATION_DIRECTORY>,
                "filenames": [<LIST_OF_FILENAMES_WITH_.PDF_IN_HOST_SERVER>,...]
            }
    '''
    import paramiko
    from scp import SCPClient
    import socket
    import glob
    import os
    import sys

    hostname = socket.gethostname()
    hostIPAddr = socket.gethostbyname(hostname)
    request = json.loads(request.body.decode('utf-8'))
    user = request['user']
    server = request['server']
    password = request['password']
    port = request['port']
    server_dirpath = request['server_dirpath']
    host_dirpath = request['host_dirpath']
    filenames = request['filenames']
    if isinstance(filenames, str):
        filenames = [filenames]

    fname = [os.path.splitext(i)[0] for i in filenames]

    def createSSHClient(server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    # Define progress callback that prints the current percentage completed for the file
    def progress(filename, size, sent):
        sys.stdout.write("%s\'s progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))

    ssh = createSSHClient(server, port, user, password)
    all_orgin_dest_pairs = []
    for name in fname:
        orgin_list = glob.glob(os.path.join(host_dirpath, f"{name}*"))
        dest_list = [os.path.join(server_dirpath, os.path.basename(i)) for i in orgin_list]
        orgin_dest_pairs = list(zip(orgin_list, dest_list))
        all_orgin_dest_pairs.extend(orgin_dest_pairs)
    success = []
    with SCPClient(ssh.get_transport(), progress=progress) as scp:
        for origin_filepath, dest_filepath in all_orgin_dest_pairs:
            scp.put(origin_filepath, dest_filepath)  # Copy my_file.txt to the server
            success.append(os.path.basename(origin_filepath))
    scp.close()
    response = {f'Files successfully transfer from host {hostIPAddr} to server {server}': success}
    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def delete_data(request):
    '''
    Delete record of uploaded pdf with document_id on database table "pdffiles"
    Prerequisite: Uploaded report pdf to data/pdf
    request = {"doc_ids": [List of "COMPANY_YEAR_eng"]}
    '''
    request = json.loads(request.body.decode('utf-8'))
    doc_id = request['doc_id']
    du = DbUtil()
    if isinstance(doc_id, str):
        print(doc_id)
        delete_dict = {
            "document_id": doc_id
        }
        du.delete_data(PDFFILES_TABLENAME, delete_dict)
    else:
        for i in doc_id:
            print(i)
            delete_dict = {
                "document_id": i
            }
            du.delete_data(PDFFILES_TABLENAME, delete_dict)
    response = {'success': True}
    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def delete_files(request):
    """
    Given a list or string of filenames (the pdf filename that exists in pdf folder),
    and a list or string of task names (must be "pdf_files", "document_parsing", "text_metric", "text_reasoning", "table_extraction", "table_detection", "pdf_annotation", "docparse_csv", "layoutlm_input", "layoutlm_output")
    delete the files or results of particular tasks

    request = {"filenames": [FILENAME.pdf, ...], "tasks": ["document_parsing",...]}
    """
    request = json.loads(request.body.decode("utf-8"))
    filenames = request.get("filenames")
    tasks = request.get("tasks")
    data_dirs_types = {
        "pdf_files": (PDF_DIR, '.pdf'),
        "document_parsing": (DOCPARSE_OUTPUT_JSON_DIR, '.json'),
        "text_metric": (METRIC_OUTPUT_JSON_DIR, '.json'),
        "text_reasoning": (REASONING_OUTPUT_JSON_DIR, '.json'),
        "table_extraction": (TABLE_OUTPUT_JSON_DIR, '.json'),
        "table_detection": (OUTPUT_TABLE_DETECTION_DIR, '.json'),
        "pdf_annotation": (
        OUTPUT_ANNOT_PDF_DIR, ['_annot_ele.pdf', '_annot_model_ele.pdf', '_annot_order.pdf', '_annot_txtblk.pdf']),
        "docparse_csv": (OUTPUT_CSV_DIR, ['_content_tokens.csv', '_model_string.csv', '_string.csv']),
        "layoutlm_input": (OUTPUT_LAYOUTLM_INPUT_DIR, '.json'),
        "layoutlm_output": (OUTPUT_LAYOUTLM_OUTPUT_DIR, '.json')
    }
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(tasks, str):
        tasks = [tasks]

    successful_delete = []
    fail_delete = []
    for task in tasks:
        try:
            t = data_dirs_types[task]
            data_dir = t[0]
            file_extension = t[1]
        except KeyError:
            e = f'The task name {task} is not acceptable. Please provide the task names among ["pdf_files", "document_parsing", "text_metric", "text_reasoning", "table_extraction", "table_detection", "pdf_annotation", "docparse_csv", "layoutlm_input", "layoutlm_output"]'
            traceback.print_exc()
            response = {
                "success": False,
                "error": str(e)
            }
            return response
        key_ts = ['TS', 'term sheet']
        key_fa = ['FA', 'facility agreement', 'facilities agreement']
        if isinstance(file_extension, str):
            file_extension = [file_extension]
        for filename in filenames:
            fname = os.path.splitext(filename)[0]
            if re.match(r'.*' + r'.*|.*'.join(key_ts), fname, flags=re.IGNORECASE):
                sub_folder = 'TS/'
            elif re.match(r'.*' + r'.*|.*'.join(key_fa), fname, flags=re.IGNORECASE):
                sub_folder = 'FA/'
            else:
                sub_folder = ''
            for ext in file_extension:
                if task in ["pdf_files", "document_parsing", "docparse_csv"]:
                    filepath = os.path.join(data_dir + sub_folder, fname + ext)
                else:
                    filepath = os.path.join(data_dir, fname + ext)
                if os.path.exists(filepath):
                    successful_delete.append(filepath)
                    os.remove(filepath)
                else:
                    fail_delete.append(filepath)

    response = {"success": True, "successfully_deleted": successful_delete, "failed_to_delete": fail_delete}

    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def download_files(request):
    """
    Given a list or string of filenames (the pdf filename that exists in pdf folder),
    and a list or string of task names (must be "pdf_files", "document_parsing", "text_metric", "text_reasoning", "table_extraction", "table_detection", "pdf_annotation", "docparse_csv", "layoutlm_input", "layoutlm_output")
    Zip and download the files or results of particular tasks in a batch

    request = {"filenames": [FILENAME.pdf, ...], "tasks": ["document_parsing",...]}
    """
    import zipfile
    from io import BytesIO

    request = json.loads(request.body.decode("utf-8"))
    filenames = request.get("filenames")
    tasks = request.get("tasks")
    data_dirs_types = {
        "pdf_files": (PDF_DIR, '.pdf', "application/pdf"),
        "document_parsing": (DOCPARSE_OUTPUT_JSON_DIR, '.json', 'application/json'),
        "text_metric": (METRIC_OUTPUT_JSON_DIR, '.json', 'application/json'),
        "text_reasoning": (REASONING_OUTPUT_JSON_DIR, '.json', 'application/json'),
        "table_extraction": (TABLE_OUTPUT_JSON_DIR, '.json', 'application/json'),
        "table_detection": (OUTPUT_TABLE_DETECTION_DIR, '.json', 'application/json'),
        "pdf_annotation" : (OUTPUT_ANNOT_PDF_DIR,['_annot_ele.pdf','_annot_model_ele.pdf','_annot_order.pdf','_annot_txtblk.pdf'],"application/pdf"),
        "docparse_csv": (OUTPUT_CSV_DIR, ['_content_tokens.csv','_model_grp_tokens.csv','_grp_tokens.csv','_child.csv' ,'_bbox.csv'],'text/csv'),
        "layoutlm_input": (OUTPUT_LAYOUTLM_INPUT_DIR, '.json', 'application/json'),
        "layoutlm_output": (OUTPUT_LAYOUTLM_OUTPUT_DIR, '.json', 'application/json')
    }
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(tasks, str):
        tasks = [tasks]

    files = []

    for task in tasks:
        try:
            t = data_dirs_types[task]
            data_dir = t[0]
            file_extension = t[1]
            content_type = t[2]
        except KeyError:
            e = f'The task name {task} is not acceptable. Please provide the task names among ["pdf_files", "document_parsing", "text_metric", "text_reasoning", "table_extraction", "table_detection", "pdf_annotation", "docparse_csv", "layoutlm_input", "layoutlm_output"]'
            traceback.print_exc()
            response = {
                    "success": False,
                    "error": str(e)
            }
            return response

        if isinstance(file_extension, str):
            file_extension = [file_extension]
        for filename in filenames:
            fname = os.path.splitext(filename)[0]
            for ext in file_extension:
                filepath = os.path.join(data_dir, fname + ext)
                if os.path.exists(filepath):
                    files.append((filepath, task))

    # Folder name in ZIP archive which contains the above files
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H%M%S")
    zip_subdir = "results_%s" % date_time
    zip_filename = "%s.zip" % zip_subdir
    # Open BytesIO to grab in-memory ZIP contents
    s = BytesIO()
    # The zip compressor
    zf = zipfile.ZipFile(s, "w")

    for fpath, task in files:
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)
        zip_path = os.path.join(zip_subdir, task + '/' + fname)

        # Add file, at correct path
        zf.write(fpath, zip_path)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    response = HttpResponse(s.getvalue(), content_type="application/x-zip-compressed")
    # ..and correct content-disposition
    response['Content-Disposition'] = 'attachment; filename=%s' % zip_filename

    return response

@sync_to_async
def generate_key_metrics(request):
    '''
    request =
    {
        "test_info": {"filename.pdf":[LIST OF PAGE ID],...},
        "filenames": [LIST OF FILENAMES.pdf],
    }
    '''
    request = json.loads(request.body.decode('utf-8'))
    test_info = request.get('test_info')
    filenames = request.get('filenames')
    mode = request.get('mode', 'TSR')
    i = 1
    if test_info:
        all_results = []
        for filename, page_numbers in test_info.items():
            try:
                i += 1
                company_name = filename.split('/')[-1].split('_')[0]
                year = min([int(i) for i in re.findall('\d+', filename, re.I) if re.match('\d{4}$', i)])

                document_id = request.get(
                    'document_id', f'{company_name}_{str(year)}')
                print(document_id)
                delete_dict = {
                    "document_id": document_id
                }
                result = main_func(filename, page_numbers, document_id, mode=mode)
                # du = DbUtil()
                # du.delete_data(METRICS_TABLENAME2, delete_dict)
                # du.insert_data(table_name=METRICS_TABLENAME2, data_list=result)
                all_results.extend(result)

            except Exception as e:
                traceback.print_exc()
                response = {'success': False, 'error': str(e)}
        response = {'success': True, 'result': all_results}
    elif filenames:
        for filename in filenames:
            try:
                t = datetime.datetime.now()
                i += 1
                company_name = filename.split('/')[-1].split('_')[0]
                year = min([int(i) for i in re.findall('\d+', filename, re.I) if re.match('\d{4}$', i)])

                document_id = request.get(
                    'document_id', f'{company_name}_{str(year)}')
                delete_dict = {
                    "document_id": document_id
                }
                doc = fitz.open(get_filepath_with_filename(PDF_DIR, filename))
                page_count = doc.page_count
                print(document_id)
                print('total pages: ', page_count)
                page_numbers = [i + 1 for i in range(page_count)]
                raw_result = main_func(filename, page_numbers, document_id, mode=mode)
                result = []
                for item in raw_result:
                    uom = item['unit']
                    target_metric = item['target_metric']
                    metric_year = item['year']
                    multiplier, converted_unit = convert_arbitrary_uom(
                        uom, target_metric, metric_year)
                    item['multiplier'] = multiplier
                    item['converted_unit'] = converted_unit
                    result.append(item)
                # du = DbUtil()
                # du.delete_data(METRICS_TABLENAME2, delete_dict)
                # du.insert_data(table_name=METRICS_TABLENAME2, data_list=result)
                print('Time Cost: ', datetime.datetime.now() - t)
                response = {'success': True, 'result': result}

            except Exception as e:
                traceback.print_exc()
                response = {'success': False, 'error': str(e)}
    else:
        filenames = []
        for filename in os.listdir('data/pdf/'):
            if filename.endswith('.pdf'):
                if not filename.startswith('0'):
                    filenames.append(filename)
        filenames = list(set(filenames))
        print('all files: ', filenames)
        for filename in filenames:
            try:
                i += 1
                company_name = filename.split('/')[-1].split('_')[0]
                year = re.findall('\d\d\d\d', filename, re.I)[0]

                document_id = request.get(
                    'document_id', f'{company_name}_{year}_en')
                print(document_id)
                # document_id = f'ESG_undefined_pages_{i}'
                delete_dict = {
                    "document_id": document_id
                }
                table_outpath = os.path.join(
                    TABLE_OUTPUT_JSON_DIR,
                    re.sub(".pdf", ".json", filename)
                )
                # if not os.path.exists(table_outpath):
                doc = fitz.open(get_filepath_with_filename(PDF_DIR, filename))
                page_count = doc.page_count
                print(filename, document_id)
                print('total pages: ', page_count)
                page_numbers = [i + 1 for i in range(page_count)]
                result = main_func(filename, page_numbers, document_id)
                with open(table_outpath, 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                du = DbUtil()
                du.delete_data(METRICS_TABLENAME2, delete_dict)
                du.insert_data(table_name=METRICS_TABLENAME2, data_list=result)
                response = {'success': True, 'result': result}
            except Exception as e:
                traceback.print_exc()
                response = {'success': False, 'error': str(e)}
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def upload_file(request):
    ''' upload pdf file to ../data/pdf & save the basic information to the database table name 'pdffiles'
    If the pdf is proceeded, return the table metric extraction results from database table "test_environment_info"
    request =
    {
        "filepath":"data/pdf/海隆控股_2020Environmental,SocialandGovernanceReport.pdf",
        "document_id": "海隆控股_2020_eng",
        "industry": "Energy"
    }
    '''
    request = json.loads(request.body.decode('utf-8'))
    document_id = request['document_id']
    filepath = request['filepath']
    filename = filepath.split('/')[-1]
    print('request:', request)
    # company = filepath.split('/')[-1].split('_')[0]
    # year = min([int(i) for i in re.findall('\d+', filename, re.I) if re.match('\d{4}$', i)])
    industry = request['industry']
    company = document_id.split('_')[0]
    year = document_id.split('_')[1]
    report_language = document_id.split('_')[-1]
    # report_type = request['type']
    if re.search(r'.*Annual Report.*', filename, re.I):
            report_type = 'annualReport'
    elif re.search(r'.*Carbon Neutrality.*', filename, re.I):
        report_type = 'carbonNeutralityReport'
    else:
        report_type = 'esgReport'
    uploaded_date = datetime.datetime.today().replace(tzinfo=pytz.utc)
    status = 'Processing'
    result = [{
        'document_id': document_id,
        'filename': filename,
        'industry': industry,
        'company': company,
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
    du = DbUtil()
    try:
        du.delete_data(PDFFILES_TABLENAME, delete_dict)
    except:
        pass
    du.insert_data(
        table_name=PDFFILES_TABLENAME,
        data_list=result
    )
    report_info = generate_report_info(filename)
    try:
        du.delete_data(PDFFILES_INFO_TABLENAME, {'filename':filename})
    except:
        pass
    du.insert_data(
        table_name=PDFFILES_INFO_TABLENAME,
        data_list=[report_info]
    )
    try:
        # process pdf & extract
        key_metrics_list, metric_er, reasoning_er = pdf_analyze(
            filename, document_id)
        du.delete_data(METRICS_TABLENAME, delete_dict)
        # assert key_metrics_list, "key_metrics_list is either None or a empty list"
        du.insert_data(table_name=METRICS_TABLENAME,
                       data_list=key_metrics_list)
        status = 'Processed'
        completed_date = datetime.datetime.today().replace(tzinfo=pytz.utc)
        filter_dict = {
            "document_id": document_id
        }
        update_dict = {
            'process1_status': 'All analysis completed',
            'process2_status': 'All analysis completed',
            'last_process_date': completed_date,
            "status": status,
        }
        du.update_data(
            PDFFILES_TABLENAME,
            filter_dict=filter_dict,
            update_dict=update_dict
        )
        result[0]['status'] = status
        print(key_metrics_list)
        response = {
            'success': True,
            'key_metrics_list': len(key_metrics_list) if key_metrics_list else None,
            'metric_entity_relations': len(metric_er) if metric_er else None,
            'reasoning_entity_relations': len(reasoning_er) if reasoning_er else None,
        }
    except Exception as e:
        # du.delete_data(PDFFILES_TABLENAME, delete_dict)
        traceback.print_exc()
        # print(filename, e)
        response = {'success': False, 'error': str(e)}
    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def update_report_info(request):
    files = os.listdir(PDF_DIR)
    for filename in files:
        report_info = generate_report_info(filename)
        du = DbUtil()
        try:
            du.delete_data(PDFFILES_INFO_TABLENAME, {'filename':filename})
        except:
            pass
        du.insert_data(
            table_name=PDFFILES_INFO_TABLENAME,
            data_list=[report_info]
        )
    response = {'success': True, 'pdf_list': files}
    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def get_uploadlist(request):
    ''' Query the uploaded list of pdf files from database table 'pdffiles' and the processing status.
    If the pdf is proceeded, return the table metric extraction results from database table "test_environment_info"
    request = {}
    '''
    du = DbUtil()
    pdf_files = du.select_table(table_name=PDFFILES_TABLENAME, field_list=[])
    df = pd.DataFrame(data=pdf_files)
    print('processed pdf files: ', list(df['document_id']))

    # df = df[df.status == 'Processed']
    df_time = df.select_dtypes(['datetimetz', 'datetime64'])

    def datetime2str(obj):
        try:
            string = datetime.datetime.strftime(obj, '%Y-%m-%d %H:%M:%S')
            return string
        except:
            return None
    df[df_time.columns] = df[df_time.columns].applymap(
        lambda i: datetime2str(i))

    result = df.groupby('document_id').max().reset_index().fillna(
        np.nan).replace([np.nan], [None]).to_dict(orient='records')
    response = {
        'success': True, 'pdf_files': len(result), 'result': result
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def get_category_options(request):
    ''' get the options for subject, target_aspect and disclosure
    request = {}
    '''
    du = DbUtil()
    metric_schema = du.select_table(
        table_name=METRIC_SCHEMA_TABLENAME, field_list=[])
    df = pd.DataFrame(data=metric_schema)
    dic = {}
    subjects = list(set([s for s in df.subject if s]))
    for subject in subjects:
        sub_df = df[df.subject == subject]
        targets = list(set([t for t in sub_df.target_aspect if t]))
        sub_dic = dict()
        for target in targets:
            sub_dic.update(
                {target: list(set(sub_df[sub_df.target_aspect == target].disclosure))})
        dic.update({subject: sub_dic})
    response = {
        'success': True, 'result': [dic]
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_industry_average(request):
    '''
    request =
        {
            "industry": "energy",
            "subject": "environmental",
            "target_aspect": "Emissions",
            "disclosure": "GHG Emissions"
        }
    '''
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    industry = request.get('industry', 'energy')
    subject = request['subject']
    target_aspect = request['target_aspect']
    disclosure = request['disclosure']

    metric_schema = du.select_table(
        table_name="metric_schema",
        filter_dict={
            "subject": subject,
            "target_aspect": target_aspect,
            "disclosure": disclosure,
            "category": "Intensity"
        }
    )
    print('metric_schema: ', metric_schema)
    flat_metric_dict = {}
    for item in metric_schema:
        if isinstance(item['unit'], list):
            unit = item['unit'][0]
        elif isinstance(item['unit'], str):
            unit = item['unit']
        for metric in ast.literal_eval(item['metric']):
            flat_metric_dict.update({metric: unit})

    # add indusry as filter later
    table_metrics_info = du.select_table(
        table_name=METRICS_TABLENAME2
    )
    df = pd.DataFrame(data=table_metrics_info)
    df = df[df.similar_score >= 0.6]
    df_sorted = df.sort_values(
        ['company_name', 'target_metric', 'similar_score', 'value', 'year', 'unit'])
    results_updated = df_sorted.drop_duplicates(
        subset=[
            'company_name',
            'target_metric',
            # 'value',
            'year',
            # 'unit'
        ]
    ).to_dict(orient='records')

    sub_info = []
    for item in results_updated:
        if item['target_metric'] in list(flat_metric_dict.keys()):
            sub_info.append({
                'target_metric': item['target_metric'],
                'value_float': item['converted_value'],
                'year': item['year'],
                'unit': flat_metric_dict.get(item['target_metric'], '')
            })
    metric_df = pd.DataFrame(data=sub_info)
    result = []
    if not metric_df.empty:
        df = metric_df.groupby(['year', 'target_metric'])[
            'value_float'].sum().reset_index()
        df['unit'] = df['target_metric'].apply(
            lambda i: flat_metric_dict.get(i, ''))
        result = df.to_dict(orient='records')
    response = {
        'success': True, 'result': result
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_top_n_companies(request):
    '''
    request =
        {
            "industry": "energy",
            "subject": "environmental",
            "target_aspect": "Emissions",
            "disclosure": "GHG emissions",
            "year": "2020"
        }
    '''
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    industry = request.get('industry', 'energy')
    subject = request['subject']
    target_aspect = request['target_aspect']
    disclosure = request['disclosure']
    year = request['year']

    metric_schema = du.select_table(
        table_name="metric_schema",
        filter_dict={
            "subject": subject,
            "target_aspect": target_aspect,
            "disclosure": disclosure,
            "category": "Intensity"
        }
    )
    flat_metric_dict = {}
    for item in metric_schema:
        if isinstance(item['unit'], list):
            unit = item['unit'][0]
        elif isinstance(item['unit'], str):
            unit = item['unit']
        for metric in ast.literal_eval(item['metric']):
            flat_metric_dict.update({metric: unit})

    # add indusry as filter later
    table_metrics_info = du.select_table(
        table_name=METRICS_TABLENAME2,
        filter_dict={
            "year": year
        }
    )
    sub_info = []
    for item in table_metrics_info:
        if item['target_metric'] in list(flat_metric_dict.keys()):
            value_float = item['converted_value']
            if value_float:
                sub_info.append({
                    'target_metric': item['target_metric'],
                    'value_float': value_float,
                    'company_name': item['company_name']
                })

    metric_df = pd.DataFrame(data=sub_info)
    result = []
    if not metric_df.empty:
        df = metric_df.groupby(['company_name', 'target_metric'])[
            'value_float'].sum().reset_index()
        df['unit'] = df['target_metric'].apply(
            lambda i: flat_metric_dict.get(i, ''))
        for metric in set(df.target_metric):
            result.append({
                metric: df[df.target_metric == metric].sort_values(['value_float'], ascending=False).to_dict(
                    orient='records')
            })
    response = {
        'success': True, 'result': result
    }
    return HttpResponse(json.dumps(response, indent=4))


def view_useful_metrics(request):
    '''
    view useful metrics about waste: hazardous waste & non-hazardous waste
    request
    {
        "year": "2020"
    }
    '''
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    year = request['year']
    disclosures = ["Hazardous Wastes", "Non-hazardous Wastes"]
    metric_schema = du.select_table(
        table_name="metric_schema",
        field_list=['disclosure', 'metric', 'category'],
        filter_dict={
            "disclosure__in": disclosures
        },
    )
    metric_dict = {}
    for item in metric_schema:
        if item['category'] != 'Intensity':
            for metric in ast.literal_eval(item['metric']):
                metric_dict[metric] = item['disclosure']

    table_metrics_info = du.select_table(
        table_name=METRICS_TABLENAME2,
        filter_dict={
            "year": year
        }
    )
    sub_info = []
    companies_with_hazardous_waste = []
    companies_with_non_hazardous_waste = []
    for item in table_metrics_info:
        if metric_dict.get(item['target_metric']):
            d = metric_dict.get(item['target_metric'])
            if d == "Hazardous Wastes":
                companies_with_hazardous_waste.append(item['company_name'])
            else:
                companies_with_non_hazardous_waste.append(item['company_name'])
            value_float = item['converted_value']
            if value_float:
                sub_info.append({
                    'target_metric': item['target_metric'],
                    'value_float': value_float,
                    'disclosure': metric_dict.get(item['target_metric'])
                })
    total_num_of_companies = len(
        set([item['company_name'] for item in table_metrics_info]))
    pct_hzd = len(set(companies_with_hazardous_waste)) / total_num_of_companies
    pct_non_hzd = len(set(companies_with_non_hazardous_waste)
                      ) / total_num_of_companies
    metric_df = pd.DataFrame(data=sub_info)
    result = []
    if not metric_df.empty:
        for disclosure in set(metric_df.disclosure):
            df_s = metric_df[metric_df.disclosure == disclosure]
            s = df_s.groupby(['target_metric'])[
                'value_float'].sum() / sum(df_s.value_float)
            result.append({
                # disclosure: s.to_dict()
                disclosure: {k: round(v, 4) for k, v in s.to_dict().items()}
            })
    response = {
        'success': True, 'result': result,
        'percentage_of_companies_with_hazardous_waste': round(pct_hzd, 4),
        'percentage_of_companies_with_non_hazardous_waste': round(pct_non_hzd, 4)
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_company_profile(request):
    '''
    request = {}
    '''
    du = DbUtil()
    metric_schema = du.select_table(
        table_name="metric_schema",
        field_list=['disclosure', 'metric', 'category', 'compulsory'],
        filter_dict={
            "compulsory": True
        }
    )
    compulsory_metrics_list = []
    for item in metric_schema:
        compulsory_metrics_list.extend(ast.literal_eval(item['metric']))
    compulsory_metrics_list = list(set(compulsory_metrics_list))

    table_metrics_info = du.select_table(
        table_name=METRICS_TABLENAME2
    )
    entity_metrics_info = du.select_table(
        table_name=METRIC_EXTRACTION_TABLE_NAME
    )
    all_info = table_metrics_info + entity_metrics_info
    df = pd.DataFrame(data=all_info)
    df['similarity'] = df['similarity'].fillna(df['similar_score'])

    cols = [
        # 'document_id',
        'company_name',
        'year',
        'target_metric',
        'metric',
        'similarity',
        'value',
        'unit',
        'similar_score',
        'converted_value',
        'converted_unit'
    ]

    df = df[cols]
    df = df[df['target_metric'].isin(compulsory_metrics_list)]
    df = df.sort_values(
        ['company_name', 'year', 'similarity'],
        ascending=False
    )
    df = df.drop_duplicates(subset=['company_name', 'year', 'target_metric'])
    df = df.fillna('')
    df['value_float'] = df['converted_value']
    result = []
    metric_unit_map = {}
    metric_df = pd.read_csv(METRIC_SCHEMA_CSV)
    for item in metric_df.to_dict(orient='records'):
        for metric in ast.literal_eval(item['metric']):
            try:
                if isinstance(item['unit'], str):
                    if re.search('\[', item['unit']):
                        metric_unit_map[metric] = ast.literal_eval(item['unit'])[0]
                    else:
                        metric_unit_map[metric] = item['unit']
                else:
                    pass
            except Exception as e:
                print(e, item['unit'])
    for name in set(df.company_name):
        sub_df = df[df.company_name == name]
        content = dict()
        for item in sub_df.to_dict(orient='records'):
            try:
                uom = item['unit']
                target_metric = item['target_metric']
                metric_year = item['year']
                # TODO: remove this part to upload function
                # multiplier, converted_unit = convert_arbitrary_uom(uom, target_metric, metric_year)
                converted_unit = metric_unit_map[target_metric]
                # if not multiplier:
                #     multiplier = 1
                content['{} ({})'.format(target_metric, converted_unit)] = item['value_float']
            except Exception as e:
                print(e, item)
        result.append({
            'company_name': name,
            'metric': [content]
        })
    response = {
        'success': True, 'result': result
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_key_performance_kpi(request):
    '''
    request =
    {
        "company_name" : ,
        "key_performance_kpi" :
    }
    '''
    request = json.loads(request.body.decode('utf-8'))
    company_name = request['company_name']
    kpi = request['key_performance_kpi']

    du = DbUtil()
    table_metrics_info = du.select_table(
        table_name=METRICS_TABLENAME2,
        filter_dict={
            "company_name": company_name
        }
    )
    # temporary, TODO: update data/use two tables in db
    df = pd.DataFrame(data=table_metrics_info)
    df = df[df.similar_score >= 0.6]
    df_sorted = df.sort_values(
        ['target_metric', 'similar_score', 'value', 'year', 'unit'])
    results_updated = df_sorted.drop_duplicates(
        subset=[
            'target_metric',
            # 'value',
            'year',
            # 'unit'
        ]
    ).to_dict(orient='records')

    kpi_list = KEY_PERFORMANCE_KPI_SCHEMA.get(kpi)
    result = []
    for item in results_updated:
        item['value_float'] = item['converted_value']
        if item['target_metric'] in kpi_list:
            result.append(item)
    response = {
        'success': True, 'result': result
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_metrics(request):
    ''' get all metric extraction results from table extraction for the processed pdf files
    request = {"document_id": "COMPANY_YEAR_eng"}
    '''
    request = json.loads(request.body.decode('utf-8'))
    document_id = request['document_id']
    ### temporary, TODO: update document_id format later filename_year_language
    # language: eng/zh
    # if re.search('eng|zh', document_id):
    #     document_id = re.sub('_eng|_zh', '', document_id)
    du = DbUtil()
    env_unit_df = pd.read_csv(METRIC_SCHEMA_CSV)
    env_dict = dict()
    for item in env_unit_df.to_dict(orient='records'):
        metrics = ast.literal_eval(item['metric'])
        for m in metrics:
            env_dict[m] = item['subject']

    table_metrics_info = du.select_table(
        table_name=METRICS_TABLENAME2,
        filter_dict={
            "document_id__in": [
                re.sub('_eng|_zh', '', document_id),
                document_id
            ]
        }
    )
    all_metrics_info = []
    for item in table_metrics_info:
        try:
            if item.get('value'):
                value_float = convert_str_to_float(item['value'])
                if value_float:
                    dic = dict()
                    dic['data_source'] = 'table'
                    dic['document_id'] = document_id
                    dic['year'] = item['year']
                    dic['company_name'] = item['company_name']
                    dic['page_no'] = item['page_no']
                    dic['metric'] = item.get('metric', '')
                    dic['target_metric'] = item['target_metric']
                    dic['subject'] = env_dict.get(item['target_metric'])
                    dic['value'] = value_float
                    dic['unit'] = item.get('unit', '')
                    dic['similarity'] = item['similar_score']
                    dic['id'] = item['id']
                    dic['type'] = 'raw'
                    all_metrics_info.append(dic)
        except Exception as e:
            traceback.print_exc()
            print(item, e)

    # modify this in production
    try:
        company_name = table_metrics_info[0]['company_name']
    except Exception as e:
        print(
            f'There is no document_id: {document_id} in database {METRICS_TABLENAME2}')
        print(e)
        company_name = document_id.split('_')[0]
    entity_metrics_info = du.select_table(
        table_name=METRIC_EXTRACTION_TABLE_NAME,
        filter_dict={
            "company_name": company_name
        }
    )
    for item in entity_metrics_info:
        try:
            if item.get('value'):
                value_float = convert_str_to_float(item['value'])
                if value_float:
                    dic = dict()
                    dic['data_source'] = 'text'
                    dic['document_id'] = document_id
                    dic['year'] = item['year']
                    dic['company_name'] = item['company_name']
                    dic['page_no'] = item['page_id']
                    dic['metric'] = item.get('metric', '')
                    dic['target_metric'] = item['target_metric']
                    dic['subject'] = env_dict.get(item['target_metric'])
                    dic['value'] = value_float
                    dic['unit'] = item.get('unit', '')
                    dic['similarity'] = item['similarity']
                    dic['id'] = item['id']
                    dic['type'] = 'raw'
                    all_metrics_info.append(dic)
        except Exception as e:
            traceback.print_exc()
            print(item, e)
    derivation = []
    derivation = gen_derived_data(all_metrics_info)

    response = {
        'success': True, 'result': all_metrics_info + derivation
    }
    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def import_update_schema(request):
    ''' Import or update a schema from a csv in 'data/schema/', either target_metric.csv for metric extraction or target_aspect.csv for reasoning
    request =
    {"schema_name": "metric"} or {"schema_name": "reasoning"}
    '''
    request = json.loads(request.body.decode("utf-8"))
    schema_name = request["schema_name"]
    if schema_name == "metric":
        update_metric_schema_from_csv(
            METRIC_SCHEMA_CSV, MetricSchema, tablename="metric_schema")
        response = {
            'success': True, 'message': "update metric schema to database table 'metric_schema' "
        }
        return HttpResponse(json.dumps(response, indent=4))
    else:
        update_target_aspect_schema_from_csv(
            TARGET_ASPECT_TABLE, TargetAspect, tablename="target_aspect")
        response = {
            'success': True, 'message': "update target aspect schema to database table 'target_aspect' "
        }
        return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_reasoning(request):
    ''' get all reasoning extraction results for the processed pdf files
    request =
    {"document_id": "COMPANY_YEAR_eng"}
    '''
    request = json.loads(request.body.decode('utf-8'))
    document_id = request['document_id']
    du = DbUtil()
    pdf_files = du.select_table(
        table_name=PDFFILES_TABLENAME,
        filter_dict={
            "document_id": document_id
        })
    filename = re.sub(".pdf", ".json", pdf_files[0]['filename'])
    with open(REASONING_OUTPUT_JSON_DIR + filename, 'r') as f:
        data = json.load(f)
    result = []
    pattern = '|'.join([f'{i}'+r'(_){0,1}\d{0,}$' for i in TARGET_ELEMENTS])
    for i in data['reasoning_entity_relations']:
        d = {}
        for k, v in data.items():  # Lv1: filename, company_name, year etc.
            if type(v) is not list:
                d[k] = v
        for k1, v1 in i.items():  # Lv2: page_id, text_block_id, element_name
            if not re.match(pattern, k1):
                d[k1] = v1
            else:
                d['block_element'] = k1
                for i2 in v1:
                    for k2, v2 in i2.items():  # Lv3: sent_id, sentence, ners, split_sentence
                        if k2 not in ["ners", "split_sentence"]:
                            d[k2] = v2
                        else:
                            d[k2] = []
                            if k2 == "ners":
                                invalid_entities = []
                                for ner in v2:
                                    head_entity_type = ner['head_entity_type']
                                    head_entity = ner['head_entity']
                                    relation = ner['relation']
                                    tail_entity_type = ner['tail_entity_type']
                                    tail_entity = ner['tail_entity']
                                    is_valid_pair = False
                                    if head_entity_type == 'target' and (
                                            (relation == 'comply_with' and tail_entity_type == 'guideline') or (
                                            relation in ['increased_by',
                                                         'reduced_by'] and tail_entity_type == 'method')):
                                        is_valid_pair = True
                                    if is_valid_pair:
                                        if head_entity in invalid_entities:
                                            invalid_entities.remove(
                                                head_entity)
                                        if tail_entity in invalid_entities:
                                            invalid_entities.remove(
                                                tail_entity)
                                        d[k2].append(ner)
                                    else:
                                        invalid_entities.append(head_entity)
                                        invalid_entities.append(tail_entity)
                            else:
                                skip_index = []
                                for index in range(len(v2)):
                                    if index in skip_index:
                                        continue
                                    if v2[index]['text'] in invalid_entities:
                                        if index >= 1 and index <= len(v2) - 2:
                                            new_text = v2[index - 1]['text'].strip() + " " + v2[index][
                                                'text'].strip() + " " + \
                                                v2[index + 1]['text'].strip()
                                            new_char_pos = [v2[index - 1]['char_position'][0],
                                                            v2[index + 1]['char_position'][1]]
                                            skip_index.append(index + 1)
                                            try:
                                                d[k2].remove(
                                                    v2[index - 1]['text'].strip())
                                            except:
                                                pass
                                        elif index == 0:
                                            new_text = v2[index]['text'].strip(
                                            ) + " " + v2[index + 1]['text'].strip()
                                            new_char_pos = [v2[index]['char_position'][0],
                                                            v2[index + 1]['char_position'][1]]
                                            skip_index.append(index + 1)
                                        elif index == len(v2) - 1:
                                            new_text = v2[index - 1]['text'].strip() + \
                                                " " + v2[index]['text'].strip()
                                            new_char_pos = [v2[index - 1]['char_position'][0],
                                                            v2[index]['char_position'][1]]
                                            try:
                                                d[k2].remove(
                                                    v2[index - 1]['text'].strip())
                                            except:
                                                pass
                                        d[k2].append(
                                            {'text': new_text, 'type': 'normal', 'char_position': new_char_pos})
                                    else:
                                        d[k2].append(v2[index])
        if d['ners']:
            result.append(d)
    response = {
        'success': True, 'result': result
    }
    return HttpResponse(json.dumps(response, indent=4))


# Single perform or extract document parsing, table metric extraction and metric entity-relation extraction, output all result as 'result.json' and to database table "metric_entity_relation"
# Prerequisite: Uploaded report pdf to data/pdf
# request = {'filenames': FILENAME}
@sync_to_async
def generate_all_results(request):
    ''' use different models to generate: table extraction results, text entity extraction & reasoning entity extraction
    request = {"filename": "FILENAME.pdf"}
    '''
    request = json.loads(request.body.decode("utf-8"))
    filename = request['filename']

    key_metrics_list, metric_er, reasoning_er = pdf_analyze(filename)
    response = {
        'success': True,
        'key_metrics_list': key_metrics_list,
        'metric_entity_relations': metric_er,
        'reasoning_entity_relations': reasoning_er
    }
    with open('result.json', 'w') as f:
        json.dump(response, f, indent=4, ensure_ascii=False)
    print('completed')
    return HttpResponse(json.dumps(response, indent=4))

@sync_to_async
def update_text_entity_relation(request):
    ''' for debug purpose
    Update UIE extraction result JSON and database table name "metric_entity_relation" with latest relation, value, unit extraction function in uie_tools/utils.py
    Prerequisite: Already performed metric entity relation extraction with API endpoint generate_text_entity_relation
    request =
    {"update_type": TYPE(metric or reasoning), "filenames": [LIST OF FILENAMES.pdf]}
    OR
    {"update_type": TYPE(metric or reasoning), "filenames": [LIST_OF_FILENAMES.pdf]}
    '''
    request = json.loads(request.body.decode("utf-8"))
    update_type = request.get("update_type")
    pdf_files = request.get("filenames")
    if isinstance(pdf_files, str):
        pdf_files = [pdf_files]
    output_dir = {"metric": METRIC_OUTPUT_JSON_DIR,
                  "reasoning": REASONING_OUTPUT_JSON_DIR}
    for pdf_file in pdf_files:
        print(f'processing: {pdf_file}')
        # load parsed doc from input json directory
        try:
            outpath = os.path.join(
                output_dir[update_type],
                re.sub(".pdf", ".json", pdf_file))
            if update_type == "metric":
                output = update_mer(pdf_file)
            else:
                output = update_rer(pdf_file)
            with open(outpath, 'w') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
            response = {
                "success": True,
                "results": output
            }
        except Exception as e:
            traceback.print_exc()
            response = {
                "success": False,
                "error": str(e),
                "filename": pdf_file
            }
            return HttpResponse(json.dumps(response, indent=4))
    if isinstance(request.get("filenames"), str):
        return HttpResponse(json.dumps(response, indent=4))
    else:
        return HttpResponse(json.dumps({"success": True}, indent=4))


@sync_to_async
def generate_text_entity_relation(request):
    """
    Single/batch perform or extract document parsing and metric entity-relation extraction, output JSON and to database table "metric_entity_relation"
    Prerequisite: Uploaded report pdf to data/pdf
    request = {'filenames': FILENAME} OR {'filenames': [LIST_OF_FILENAMES]}
    example request =
    {
        "model_version": "v1",
        "filenames": [
            "PERSTA_Environmental,SocialandGovernanceReport2020.pdf",
            "PERSTA_Environmental,SocialandGovernanceReport2021.pdf",
            "上海石油化工股份_2020CorporateSocialResponsibilityReport.pdf",
            "上海石油化工股份_2021CorporateSocialResponsibilityReport.pdf",
            "中信資源_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf",
            "中信資源_2021ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf",
            "中國海洋石油_2020Environmental,SocialandGovernanceReport.pdf",
            "中國海洋石油_2021Environmental,SocialandGovernanceReport.pdf",
            "中國石油化工股份_2020SinopecCorp.SustainabilityReport.pdf",
            "中國石油化工股份_2021SinopecCorp.SustainabilityReport.pdf",
            "中國石油股份_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf",
            "中國石油股份_2021ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf",
            "中國神華_2020Environmental,ResponsibilityandGovernanceReport.pdf",
            "中國神華_2021Environmental,ResponsibilityandGovernanceReport.pdf",
            "中煤能源_ChinaCoalEnergyCSRReport2020.pdf",
            "中煤能源_ChinaCoalEnergyCSRReport2021.pdf",
            "中石化油服_2020Environmental,Social,andGovernance(ESG)Report.pdf",
            "中石化煉化工程_2020Environmental,SocialandGovernanceReport.pdf",
            "中石化煉化工程_2021Environmental,SocialandGovernanceReport.pdf",
            "中能控股_Environmental,SocialandGovernanceReport2020.pdf",
            "中能控股_Environmental,SocialandGovernanceReport2021.pdf",
            "元亨燃氣_Environmental,socialandgovernancereport2020_21.pdf",
            "兗煤澳大利亞_ESGReport2020.pdf",
            "兗煤澳大利亞_ESGReport2021.pdf",
            "兗礦能源_SocialResponsibilityReport2020OfYanzhouCoalMiningCompanyLimited.pdf",
            "匯力資源_2020Environmental,SocialandGovernanceReport.pdf",
            "匯力資源_2021Environmental,SocialandGovernanceReport.pdf",
            "南南資源_Environmental,SocialandGovernanceReport2020_21.pdf",
            "安東油田服務_2020SUSTAINABILITYREPORT.pdf",
            "安東油田服務_2021SUSTAINABILITYREPORT.pdf",
            "山東墨龍_2020Environmental,SocialandGovernanceReport.pdf",
            "山東墨龍_2021Environmental,SocialandGovernanceReport.pdf",
            "巨濤海洋石油服務_Environmental,SocialandGovernanceReport2020.pdf",
            "巨濤海洋石油服務_Environmental,SocialandGovernanceReport2021.pdf",
            "延長石油國際_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf",
            "延長石油國際_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2021.pdf",
            "惠生工程_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf",
            "惠生工程_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2021.pdf",
            "新海能源_Environmental,SocialandGovernanceReportforYear2020.pdf",
            "易大宗_Environmental,SocialandGovernanceReport2020.pdf",
            "易大宗_Environmental,SocialandGovernanceReport2021.pdf",
            "海隆控股_2020Environmental,SocialandGovernanceReport.pdf",
            "海隆控股_2021Environmental,SocialandGovernanceReport.pdf",
            "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf",
            "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2021.pdf",
            "西伯利亞礦業_Environmental,SocialandGovernanceReport2020.pdf",
            "西伯利亞礦業_Environmental,SocialandGovernanceReport2021.pdf",
            "金泰能源控股_Environmental,SocialandGovernanceReport2020.pdf",
            "金泰能源控股_Environmental,SocialandGovernanceReport2021.pdf",
            "陽光油砂_2020Environmental,SocialandGovernanceReport.pdf",
            "陽光油砂_2021Environmental,SocialandGovernanceReport.pdf",
            "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf",
            "飛尚無煙煤_2021Environmental,SocialandGovernanceReport.pdf",
            "PERSTA_Environmental,SocialandGovernanceReport2019.pdf",
            "上海石油化工股份_2019CorporateSocialResponsibilityReport.pdf",
            "中信資源_2019ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf",
            "中國石油化工股份_2019SinopecCorp.CommunicationonProgressforSustainableDevelopment.pdf",
            "中國石油股份_2019ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf",
            "中國神華_2019EnvironmentalSocialResponsibilityandCorporateGovernanceReport.pdf",
            "中煤能源_ChinaCoalEnergyCSRReport2019.pdf",
            "中石化油服_2019Environmental,Social,andGovernance(ESG)Report.pdf",
            "中石化煉化工程_2019Environmental,SocialandGovernanceReport.pdf",
            "中能控股_Environmental,SocialandGovernanceReport2019.pdf",
            "元亨燃氣_Environmental,socialandgovernancereport2019_20.pdf",
            "兗煤澳大利亞_2019Environmental,SocialandGovernanceReport.pdf",
            "兗礦能源_SocialResponsibilityReport2019OfYanzhouCoalMiningCompanyLimited.pdf",
            "匯力資源_2019Environmental,SocialandGovernanceReport.pdf",
            "南南資源_Environmental,SocialandGovernanceReport2019_20.pdf",
            "安東油田服務_2019SUSTAINABILITYREPORT.pdf",
            "巨濤海洋石油服務_Environmental,SocialandGovernanceReport2019.pdf",
            "延長石油國際_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2019.pdf",
            "惠生工程_CORPORATESOCIALRESPONSIBILITYREPORT2019.pdf",
            "新海能源_Environmental,SocialandGovernanceReportforYear2019.pdf",
            "易大宗_Environmental,SocialandGovernanceReport2019.pdf",
            "泰山石化_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2019.pdf",
            "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2019.pdf",
            "融信資源_Environmental,Social,GovernanceReport2019.pdf",
            "西伯利亞礦業_Environmental,SocialandGovernanceReport2019.pdf",
            "金泰能源控股_Environmental,SocialandGovernanceReport2019.pdf",
            "陽光油砂_2019Environmental,SocialandGovernanceReport.pdf",
            "飛尚無煙煤_2019Environmental,SocialandGovernanceReport.pdf"
        ]
    }
    """

    request = json.loads(request.body.decode("utf-8"))
    pdf_files = request.get("filenames")
    try:
        model_version = request.get("model_version")
    except:
        model_version = 'v1'
    if isinstance(pdf_files, str):
        pdf_files = [pdf_files]

    for pdf_file in pdf_files:
        print(f'processing: {pdf_file}')
        inpath = get_filepath_with_filename(DOCPARSE_OUTPUT_JSON_DIR, re.sub(".pdf", ".json", pdf_file))

        # load parsed doc from input json directory
        try:
            if not os.path.exists(inpath):
                document_parser(pdf_file)
        except Exception as e:
            traceback.print_exc()
            response = {
                "success": False,
                "error": str(e),
                "filename": pdf_file,
                "process": 'Document Parsing'
            }
            return HttpResponse(json.dumps(response, indent=4))

        # if metric extraction doesn't exists in data/text_metric_json, do metric entity relation extraction
        try:
            outpath = os.path.join(
                METRIC_OUTPUT_JSON_DIR,
                re.sub(".pdf", ".json", pdf_file))
            outpath2 = os.path.join(
                REASONING_OUTPUT_JSON_DIR,
                re.sub(".pdf", ".json", pdf_file))
            if not os.path.exists(outpath) or not os.path.exists(outpath2):
                if model_version != 'v1':
                    URL_METRIC = URL_METRIC_V3
                else:
                    URL_METRIC = URL_METRIC_V1
                output, output2 = get_entity_relation(
                    pdf_file, URL_METRIC, URL_REASONING, model_version=model_version)
                with open(outpath, 'w') as f:
                    json.dump(output, f, indent=4, ensure_ascii=False)
                with open(outpath2, 'w') as f2:
                    json.dump(output2, f2, indent=4, ensure_ascii=False)
            else:
                with open(outpath) as f:
                    output = json.load(f)
                with open(outpath2) as f2:
                    output2 = json.load(f2)
            response = {
                "success": True,
                "results": {'metric_entity_relations': output, 'reasoning_entity_relations': output2}
            }
        except Exception as e:
            traceback.print_exc()
            response = {
                "success": False,
                "error": str(e),
                "filename": pdf_file,
                "process": 'Entity Relation Extraction'
            }
            return HttpResponse(json.dumps(response, indent=4))

    if isinstance(request.get("filenames"), str):
        return HttpResponse(json.dumps(response, indent=4))
    else:
        return HttpResponse(json.dumps({"success": True}, indent=4))


@sync_to_async
def unify_uom(request):
    ''' Test UOM unification function and return multiplier and converted unit if success
    Example:
    {
        "uom": "hundred tonnes",
        "target_metric" : "Nitrogen Oxide",
        "metric_year": "2020"
    }
    return
    {
        "multiplier: 100,
        "converted_unit": "tonnes"
    }
    '''
    request = json.loads(request.body.decode("utf-8"))
    try:
        uom = request.get("uom")
    except Exception as e:
        response = {
            "success": False,
            "error": str(e),
            "message": "You must provide the unit of measurement that going to be converted."
        }
        return HttpResponse(json.dumps(response, indent=4))
    try:
        target_metric = request.get("target_metric")
        if not MetricSchema.objects.filter(
                metric__icontains=target_metric).exists():
            response = {
                "success": False,
                "message": f"{target_metric} doesn't exist in the existing metric schema. You must provide the target metric that exists in the metric schema."
            }
            return HttpResponse(json.dumps(response, indent=4))
    except Exception as e:
        response = {
            "success": False,
            "error": str(e),
            "message": "You must provide the target metric"
        }
        return HttpResponse(json.dumps(response, indent=4))
    try:
        metric_year = request.get("metric_year")
    except Exception as e:
        response = {
            "success": False,
            "error": str(e),
            "message": "You must provide the year of metric that described in the document."
        }
        return HttpResponse(json.dumps(response, indent=4))
    try:
        multiplier, converted_unit = convert_arbitrary_uom(
            uom, target_metric, metric_year)
        response = {
            "success": True,
            "multiplier": multiplier,
            "converted_unit": converted_unit,
        }
        return HttpResponse(json.dumps(response, indent=4))
    except Exception as e:
        traceback.print_exc()
        response = {
            "success": False,
            "error": str(e)
        }
        return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def get_compliance_checking_list(request):
    '''Output compliance checking list 
    Example:
    request =
        {
            "company_name": "新海能源"
        }

    return = 
    {
    "success": true,
    "result": [
    {
    "id": "1617",
    "subject": "Environmental",
    "target_aspect": "Emissions",
    "disclosure": "Air Emissions and Pollutants",
    "intensity_group": NaN,
    "2019": false,
    "2020": false
    },...
    '''
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    try:
        company_name = request['company_name']
    except Exception as e:
        response = {
            "success": False,
            "error": str(e),
            "message": "You must provide the name of company for compliance checking."
        }
        return HttpResponse(json.dumps(response, indent=4))

    company_info = du.select_table(
        table_name=METRIC_EXTRACTION_TABLE_NAME,
        filter_dict={
            "company_name": company_name
        }
    )
    metric_schema = du.select_table(
        table_name=METRIC_SCHEMA_TABLENAME,
        filter_dict={
            "compulsory": True
        }
    )
    pdf_info = du.select_table(
        table_name=PDFFILES_TABLENAME,
        filter_dict={
            "company": company_name
        }
    )
    df_metric = pd.DataFrame(data=metric_schema)
    df_com = pd.DataFrame(data=company_info)
    df_pdf = pd.DataFrame(data=pdf_info)
    if df_pdf.empty or df_com.empty:
        response = {
            "success": False,
            "message": "There is no record for this company."
        }
        return HttpResponse(json.dumps(response, indent=4))
    dic = []
    years = list(set([s for s in df_com.year if s]))
    ids = list(set([s for s in df_metric.id if s]))
    pdf_years = list(set([s for s in df_pdf.year if s]))
    for id in ids:
        sub_df_metric = df_metric[df_metric.id == id]
        subject = sub_df_metric.iloc[0]['subject']
        # print(subject)
        target_aspect = sub_df_metric.iloc[0]['target_aspect']
        disclosure = sub_df_metric.iloc[0]['disclosure']
        metric = sub_df_metric.iloc[0]['metric']
        if pd.isna(sub_df_metric.iloc[0]['intensity_group']):
            intensity_group = ''
        else:    
            intensity_group = sub_df_metric.iloc[0]['intensity_group']
        sub_dic = {
        'id': str(id),
        'subject': subject,
        'target_aspect': target_aspect,
        'disclosure': disclosure,
        'intensity_group': intensity_group
        }
        for year in years:
            sub_df = df_com[df_com.year == year]
            #id, subject, target_aspect, disclosure, metric, compulsory, intensity_group
            #target_metric in df_com = metric in df_metric
            compulsory = False
            for j in range(len(sub_df)):
                check_metric = sub_df.iloc[j]['target_metric']
                if check_metric:
                    if sub_df.iloc[j]['target_metric'] in metric:
                        compulsory = True
                        break
            sub_dic.update({year: compulsory})
        com_year = set(years)
        for year in pdf_years:
            if year not in com_year:
                compulsory = 'Not processed'
                sub_dic.update({year: compulsory})
        dic.append(sub_dic)
    #print(dic)
    response = {
        "success": True,
        "result": dic
    }

    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_industries_average(request) -> HttpResponse:
    """ Pass request using POST method. Request content is saved in request body.
    Args:
        request: Request.
            {
                "industries": ["Energy", "Properties & Construction"],
                "subject": "environmental",
                "target_aspect": "Emissions",
                "disclosure": "GHG Emissions"
            }
    Returns:
        HttpResponse:
        [{
        "industry": "Energy",
        "year": "2021",
        "metric": "Total GHG Emissions Per Employee",
        "unit": "tonnes/employee",
        "value": 6.19
        },
        {
        "industry": "Energy",
        "year": "2021",
        "metric": "Total GHG Emissions Per Production",
        "unit": "tonnes/production",
        "value": 13.27195
        }]
    """
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    industries: list = request["industries"]
    subject: str = request["subject"]
    target_aspect: str = request["target_aspect"]
    disclosure: str = request["disclosure"]

    # Get the metrics with their units
    raw_metrics_with_unit = du.select_table(
        table_name='metric_schema',
        field_list=['unit', 'metric'],
        filter_dict={
            "subject": subject,
            "target_aspect": target_aspect,
            "disclosure": disclosure,
            "category": "Intensity"
        }
    )
    metrics_with_unit = {}
    for item in raw_metrics_with_unit:
        if isinstance(item['unit'], list):
            unit = item['unit'][0]
        elif isinstance(item['unit'], str):
            unit = item['unit']
        else:
            unit = 'NULL'
        for metric in json.loads(item['metric']):
            metrics_with_unit.update({metric: unit})

    # Get Firms by Industries
    raw_company_list_data = du.select_table(
        table_name='company_industry',
        filter_dict={
            'industry__in': industries,
        },
        field_list=['stock_id', 'company_name_ch', 'company_name_en', 'industry']
    )
    # Temp code for demo
    company_industry = pd.DataFrame(raw_company_list_data)
    company_industry = company_industry.melt(id_vars='industry', value_vars=['stock_id', 'company_name_en', 'company_name_ch'], var_name='identifier')
    company_industry = company_industry[['value', 'industry']]
    company_industry.columns = ['identifier', 'industry']

    # Get Metric data
    raw_metrics_data = du.select_table(
        table_name=METRICS_TABLENAME2,
        filter_dict={
            'similar_score__gte': 0.6,
        },
        field_list=['company_name', 'year', 'target_metric', 'value']
    )
    metrics = pd.DataFrame(raw_metrics_data)
    metrics.columns = ['identifier', 'year', 'metric', 'value']
    metrics.identifier = metrics.identifier.apply(lambda x: int(x) if x.isdigit() else x)
    metrics.value = metrics.value.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    metrics = metrics.dropna()

    industries_data = metrics.merge(company_industry, how='inner')
    industries_data['unit'] = industries_data['metric'].map(metrics_with_unit)
    industries_data.dropna()
    industries_data = industries_data.groupby(['industry', 'year', 'metric', 'unit'], as_index=False).mean(['value'])

    pre_json = industries_data.to_dict(orient='records')
    if pre_json:
        response = {
            "success": True,
            "result": pre_json
        }
    else:
        response = {
            "success": False,
            "result": 'No data found.'
        }

    return HttpResponse(json.dumps(response, indent=4))


@sync_to_async
def view_nested_metrics(request) -> HttpResponse:
    """ Get Nested json for metric categories
    Method: GET
    """
    du = DbUtil()
    metrics = du.select_table(
        table_name='metric_schema',
        field_list=['subject_no', 'subject', 'aspect_no', 'target_aspect', 'disclosure', 'disclosure_no'],
        # Unknown Category, Category C doesn't including for containing too much non-digit value
        filter_Q=~ (Q(aspect_no='S') | Q(subject_no='C'))
    )
    temp = pd.DataFrame(metrics).drop_duplicates()
    temp.columns = ['subject_no', 'subject', 'aspect_no', 'aspect', 'disclosure', 'disclosure_no']
    pre_json = df_to_json(temp)

    result = {}
    if pre_json:
        result.update({'success': True})
        result.update({'result': pre_json})
    else:
        result.update({'success': False})
        result.update({'result': 'No data'})

    return HttpResponse(json.dumps(result, indent=4))


@sync_to_async
def view_nested_company_profile_with_pdf(request) -> HttpResponse:
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    industries: list = request['industries']

    # Get company profile of selected industry
    company_profile = du.select_table(
        table_name='company_industry',
        field_list=['stock_id', 'company_name_en', 'company_name_ch'],
        filter_dict={
            'industry__in': industries,
        }
    )
    company_profile = pd.DataFrame(company_profile)
    company_profile.columns = ['stock_id', 'company_name_en', 'company_name_ch']
    stock_id_list = company_profile['stock_id'].tolist()
    company_name_en = company_profile['company_name_en'].tolist()
    company_name_ch = company_profile['company_name_ch'].tolist()

    # FROM UPLOADED files, Get pdf info of selected industry companies
    # noinspection PyBroadException
    try:
        # FROM database files, Get pdf info of selected industry companies
        pdf_info1 = du.select_table(
            table_name='pdffiles_info',
            field_list=['filename', 'report_year', 'report_type', 'stock_id', 'company_name'],
            filter_dict={
                'exist_pdf__exact': True,
            },
            filter_Q=Q(stock_id__in=stock_id_list) | Q(company_name__in=company_name_en) | Q(company_name__in=company_name_ch) | Q(company_name__in=stock_id_list)
        )
        pdf_info1 = pd.DataFrame(pdf_info1)
        result = pdf_info1.to_dict(orient='records')
        response = {
            'success': True,
            'result': result
        }
    except Exception as e:
        response = {'success': False,
                    'result': str(e)
                    }
    return HttpResponse(json.dumps(response))


@sync_to_async
def view_general_company_files_by_industries(request) -> HttpResponse:
    du = DbUtil()
    request = json.loads(request.body.decode('utf-8'))
    industries: list = request['industries']

    try:
        # Get company profile of selected industry
        company_profile = du.select_table(
            table_name='company_industry',
            field_list=['company_name_en', 'company_name_ch', 'industry'],
            filter_dict={
                'industry__in': industries,
            }
        )
        company_profile = pd.DataFrame(company_profile)
        company_profile.columns = ['company_name_en', 'company_name_ch', 'industry']
        # Save company english name and chinese name to one column with melt
        company_profile = pd.melt(company_profile, id_vars=['industry'],
                                  value_vars=['company_name_en', 'company_name_ch'], var_name='name_type',
                                  value_name='company_name')
        company_profile = company_profile.drop(['name_type'], axis=1)

        pdf_file_info = du.select_table(
            table_name='pdffiles_info',
            field_list=['filename', 'company_name', 'report_year', 'report_type'],
            filter_dict={
                'exist_pdf__exact': True,
            },
            filter_Q=Q(company_name__in=company_profile['company_name'].tolist())
        )
        pdf_file_info = pd.DataFrame(pdf_file_info)
        pdf_file_info.columns = ['filename', 'company_name', 'report_year', 'report_type']
        pdf_file_info['document_id'] = pdf_file_info['company_name'] + '_' + pdf_file_info['report_year'].astype(str)
        # Add industry column
        result = pdf_file_info.merge(company_profile, how='inner', on='company_name')
        result = result.drop_duplicates()
        result = result.to_dict(orient='records')
        response = {
            'success': True,
            'result': result
        }
    except Exception as e:
        result = str(e)
        response = {
            'success': False,
            'result': result
        }
    return HttpResponse(json.dumps(response, indent=4))
