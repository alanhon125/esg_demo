import os
import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForTokenClassification
)
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets import Dataset, load_dataset
import gc
from api.config import *
from utils import *
from postgresql_storage.db_util import DbUtil
import re
import numpy as np
import json
from asgiref.sync import sync_to_async


def record_status_update(tablename, filter_dict, update_dict):
    du = DbUtil()
    du.update_data(
        tablename,
        filter_dict=filter_dict,
        update_dict=update_dict
    )


@sync_to_async
def record_progress_update(tqdm_obj, tablename, filter_dict, progress_name, message, proportion):
    '''
    update the progress of model inference by batch count
    @param tqdm_obj: tqdm decorator object
    @type tqdm_obj: class tqdm.tqdm
    @param tablename: database table name for the record
    @type tablename: str
    @param filter_dict: dictionary of filename key "filename" value "XXX.pdf"
    @type filter_dict: dict
    @param progress_name: field name of the progress in db table "tablename"
    @type progress_name: str
    @param message: message to display in the update status
    @type message: str
    @param proportion: proportion of the subprocess that contribute in the whole process
    @type proportion: float
    '''
    import re
    import datetime
    import pytz

    process_id = re.search(r'\d+', progress_name).group()
    du = DbUtil()
    process_progress = du.select_table(table_name=tablename, field_list=[progress_name], filter_dict=filter_dict)
    if process_progress:
        process_progress = process_progress[0][progress_name]
    else:
        filename = filter_dict["filename"]

        if filename[0].isdigit():
            stock_id, company_name, _ = filename.split('_')
            stock_id = int(stock_id)
            year = min([int(i) for i in re.findall('\D(20\d{2})\D?', filename, re.I)])
        else:
            if len(filename.split('_')) == 2:
                company_name, _ = filename.split('_')
                stock_id = None
            elif len(filename.split('_')) == 3 and filename.split('_')[1][0].isdigit():
                company_name, stock_id, _ = filename.split('_')
                stock_id = int(stock_id)
            elif len(filename.split('_')) == 3 and not filename.split('_')[1][0].isdigit():
                company_name, _, _ = filename.split('_')
                stock_id = None
            else:
                company_name = None
                stock_id = None
            try:
                year = min([int(i) for i in re.findall('\d+', filename, re.I) if re.match('\d{4}$', i)])
            except:
                year = None

        if re.search(r'.*Annual Report.*', filename, re.I):
            report_type = 'annualReport'
        elif re.search(r'.*Carbon Neutrality.*', filename, re.I):
            report_type = 'carbonNeutralityReport'
        else:
            report_type = 'esgReport'

        report_language = 'eng'
        try:
            industry = du.select_table(table_name='company_industry', field_list=['industry'],
                                       filter_dict={'company_name_en': company_name})[0]['industry']
        except:
            industry = None
        document_id = company_name + '_' + str(year) + '_' + report_language

        uploaded_date = datetime.datetime.today().replace(tzinfo=pytz.utc)
        status = 'Processing'
        result = [{
            'document_id': document_id,
            'filename': filename,
            'industry': industry,
            'company': company_name,
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
        try:
            du.delete_data(PDFFILES_TABLENAME, delete_dict)
        except:
            pass
        du.insert_data(
            table_name=PDFFILES_TABLENAME,
            data_list=result
        )
        process_progress = 0

    update_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
    batch_id = tqdm_obj.format_dict['n'] + 1
    batch_total = tqdm_obj.format_dict['total']
    update_dict = {
        f'process{process_id}_status': f'{message} ({str(batch_id)}/{str(batch_total)})',
        f'process{process_id}_progress': process_progress + (1 / batch_total) * proportion,
        f'last_process{process_id}_elapsed_time': tqdm_obj.format_dict['elapsed'],
        f'process{process_id}_update_date': update_datetime
    }
    record_status_update(tablename, filter_dict, update_dict)


def remove_duplicates(x):
    return list(dict.fromkeys(x))


def token_classification(data, model_path, fname, gpu_ids="0,1,2,3", batch_size=16, document_type='esgReport'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU IDs will be ordered by pci bus IDs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ[
        "TOKENIZERS_PARALLELISM"] = "false"  # disable the parallelism to avoid any hidden deadlock that would be hard to debug

    filename = fname + '.pdf'
    filename_dict = {'filename': filename}
    gpu_ids = gpu_ids.split(',')

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu_ids[0]}' if use_cuda else 'cpu')

    try:
        model_version = re.match(r'\S+(v\d)', model_path).groups()[0]
    except:
        model_version = None

    device_ids = [int(i) for i in gpu_ids]

    # accept data input either as local path, dataset repository on the Hub, dictionary or pandas Dataframe object
    if isinstance(data, str):
        if data.endswith('.json'):
            datasets = load_dataset('json', data_files=[data])
            datasets = datasets['train']
        elif data.endswith('.csv'):
            datasets = load_dataset('csv', data_files=[data])
            datasets = datasets['train']
        else:
            try:
                datasets = load_dataset(data)
                datasets = datasets['train']
            except:
                raise Exception("The data path must be end with .json, .csv or a valid dataset repository on the Hub")
    elif isinstance(data, dict):
        datasets = Dataset.from_dict(data)
    elif isinstance(data, list):
        datasets = Dataset.from_pandas(pd.DataFrame(data=data))
    elif isinstance(data, pd.DataFrame):
        datasets = Dataset.from_pandas(data)

    column_names = datasets.column_names
    # features = datasets.features
    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        labels = list(unique_labels)
        labels = [l for l in labels if l]
        labels.sort()
        return labels

    label_column_name = "ner_tags"
    # labels = get_label_list(dataset[label_column_name])
    if document_type == 'esgReport':
        labels = ['caption', 'figure', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
    elif document_type in ['agreement', 'termSheet']:
        labels = ['caption', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
    else:
        labels = ['abstract', 'author', 'caption', 'date', 'equation', 'figure', 'footer', 'list', 'paragraph',
                  'reference', 'section', 'table', 'title']
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        input_size=224
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        add_prefix_space=True,
        apply_ocr=False
    )

    # we need to define custom features
    if model_version == 'v2' or model_version is None:
        features = Features({
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(ClassLabel(names=labels)),
            'offset_mapping': Array2D(dtype="int64", shape=(512, 2)),
            'id': Sequence(feature=Value(dtype="int64"))
        })
    elif model_version == 'v3':
        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
            'offset_mapping': Array2D(dtype="int64", shape=(512, 2)),
            'id': Sequence(feature=Value(dtype="int64"))
        })

    def preprocess_data(examples):

        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        words = examples['tokens']
        boxes = examples['bboxes']
        word_labels = examples['ner_tags']

        word_labels_ids = [[label2id[i] for i in k] for k in word_labels]
        label_lengths = [len(k) for k in word_labels]

        doc_ids = [[int(id)] * label_lengths[i] for i, id in enumerate(examples['id'])]

        encoded_inputs = processor(images,
                                   words,
                                   boxes=boxes,
                                   word_labels=word_labels_ids,
                                   padding="max_length",
                                   max_length=512,
                                   stride=128,
                                   return_offsets_mapping=True,
                                   return_overflowing_tokens=True,
                                   truncation=True)

        encoding_for_doc_ids = processor(images,
                                         words,
                                         boxes=boxes,
                                         word_labels=doc_ids,
                                         padding="max_length",
                                         max_length=512,
                                         stride=128,
                                         return_offsets_mapping=True,
                                         return_overflowing_tokens=True,
                                         truncation=True)

        overflow_to_sample_mapping = encoded_inputs.pop('overflow_to_sample_mapping')
        encoded_inputs['id'] = encoding_for_doc_ids['labels']

        # change the shape of pixel values
        x = []
        for i in range(0, len(encoded_inputs['pixel_values'])):
            x.append(encoded_inputs['pixel_values'][i])
        x = np.stack(x)
        encoded_inputs['pixel_values'] = x

        # for k,v in encoded_inputs.items():
        #     print(k,np.array(v).shape)

        return encoded_inputs

    input_dataset = datasets.map(preprocess_data,
                                 batched=True,
                                 remove_columns=remove_columns,
                                 features=features)

    # Finally, let's set the format to PyTorch, and place everything on the GPU:

    input_dataset.set_format(type="torch", device=device)

    # Next, we create corresponding dataloaders.

    inference_dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False)

    if use_cuda:
        model = DataParallel(model, device_ids=device_ids)

    model.to(device)
    # Evaluation
    # put model in evaluation mode
    model.eval()
    all_predictions = []
    all_true = []
    all_bboxes = []
    all_tokens = []
    all_ids = []

    t = tqdm.tqdm(inference_dataloader, desc='LayoutLM model inference')
    print('model used: ', os.path.basename(model_path))

    image_column_name = "image_path"
    text_column_name = "tokens"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"

    for batch_idx, batch in enumerate(t):
        record_progress_update(t, PDFFILES_TABLENAME, filename_dict, 'process2_progress',
                               'Document parsing is processing ... ', 0.89)
        with torch.no_grad():
            page_ids = batch.pop('id')
            page_ids = page_ids.squeeze().tolist()

            offset_mapping = batch.pop('offset_mapping')
            offset_mapping = offset_mapping.squeeze().tolist()

            if (len(offset_mapping) == 512):
                is_subwords = [np.array(offset_mapping)[:, 0] != 0]
            else:
                is_subwords = [np.array(i)[:, 0] != 0 for i in offset_mapping]

            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward pass
            outputs = model(**batch)

            # predictions
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = batch['bbox'].squeeze().tolist()
            labels = batch['labels'].squeeze().tolist()
            input_ids = batch['input_ids'].squeeze().tolist()

            if (len(token_boxes) == 512):
                predictions = [predictions]
                token_boxes = [token_boxes]
                labels = [labels]
                input_ids = [input_ids]
                page_ids = [page_ids]

            def indices(lst, item):
                return [i for i, x in enumerate(lst) if x == item]

            def trim_list_by_indices(lst, indices):
                return [x for i, x in enumerate(lst) if i not in indices]

            flat_tokens = []
            flat_predictions = []
            flat_labels = []
            flat_boxes = []
            flat_page_ids = []

            for i, ids in enumerate(input_ids):
                prev_token = ''
                for j, id in enumerate(ids):
                    p = predictions[i][j]
                    l = labels[i][j]
                    box = token_boxes[i][j]
                    page_id = page_ids[i][j]

                    if is_subwords[i][j] and flat_tokens:
                        flat_tokens.pop(-1)
                        token = prev_token + processor.tokenizer.decode(id)
                        token = token.strip()
                        flat_tokens.append(token)
                    else:
                        token = processor.tokenizer.decode(id).strip()
                    if l != -100:
                        flat_tokens.append(token)
                        flat_predictions.append(id2label[p])
                        flat_labels.append(id2label[l])
                        flat_boxes.append(box)
                        flat_page_ids.append(page_id)
                    prev_token = token

            # remove weird character that cannot decode correctly
            weird_char_indices = indices(flat_tokens, '�')
            weird_char_indices2 = [flat_tokens.index(t) for t in flat_tokens if t.startswith('<s>')]
            list_remove_indices = weird_char_indices + weird_char_indices2
            list_remove_indices.sort()
            sanitized_tokens = trim_list_by_indices(flat_tokens, list_remove_indices) # remove weird character � in the list
            sanitized_tokens = list(map(lambda x: str.replace(x, "�", ""), sanitized_tokens)) # remove weird character � in each string
            sanitized_predictions = trim_list_by_indices(flat_predictions, list_remove_indices)
            sanitized_labels = trim_list_by_indices(flat_labels, list_remove_indices)
            sanitized_boxes = trim_list_by_indices(flat_boxes, list_remove_indices)
            sanitized_page_ids = trim_list_by_indices(flat_page_ids, list_remove_indices)

            all_ids.extend(sanitized_page_ids)
            all_predictions.extend(sanitized_predictions)
            all_true.extend(sanitized_labels)
            all_tokens.extend(sanitized_tokens)
            all_bboxes.extend(sanitized_boxes)

            del batch
            gc.collect()
            torch.cuda.empty_cache()

    # remove duplicate tokens which have same page id, prediction, true label, token and bbox
    all_bboxes = [tuple(x) for x in all_bboxes]  # turn all list of bboxes into tuple of bboxes
    all_res = list(zip(all_ids, all_predictions, all_true, all_tokens, all_bboxes))
    all_res = remove_duplicates(all_res)
    all_res = list(zip(*all_res))
    all_ids, all_predictions, all_true, all_tokens, all_bboxes = all_res

    data = {
        "id": all_ids,
        "tokens": all_tokens,
        "bboxes": all_bboxes,
        "ner_tags": all_true,
        "predictions": all_predictions
    }
    with open("{}{}{}".format(OUTPUT_LAYOUTLM_OUTPUT_DIR, fname, '.json'), 'w') as output:
        json.dump(data, output, ensure_ascii=False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = list(zip(all_tokens, all_bboxes, all_ids, all_true, all_predictions))
    keys = ['token', 'bbox', 'page_id', 'rule_tag', 'tag']
    result = [dict(zip(keys, i)) for i in result]

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tuned Layoutlmv2 with ESG reports or agreement inference')

    parser.add_argument(
        '--data',
        required=True,
        help='Input data in pandas DataFrame or dictionary',
    )

    parser.add_argument(
        '--model_path',
        default='models/checkpoints/layoutlm/layoutlmv3_large_500k_docbank_epoch_1_lr_1e-5_1407_esg_epoch_2000_lr_1e-5',
        type=str,
        required=False,
        help='The directory store fine-tuned model checkpoint and log file',
    )

    parser.add_argument(
        '--fname',
        type=str,
        required=True,
        help='The filename of the document',
    )

    parser.add_argument(
        '--gpu_ids',
        default='0,1,2,3',
        type=str,
        required=False,
        help='The GPU IDs utilize for model inference',
    )

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        required=False,
        help='Batch size of model inference',
    )

    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        required=False,
        help='Batch size of model inference',
    )

    parser.add_argument(
        "--document_type",
        default=DOCPARSE_DOCUMENT_TYPE,
        type=str,
        required=False,
        help="Document type that pdf belongs to. Either 'esgReport' , 'agreement' or 'termSheet'",
    )

    args = parser.parse_args()
    all_predictions = token_classification(args.data, args.model_path, args.fname, gpu_ids=args.gpu_ids,
                                           batch_size=args.batch_size, document_type=args.document_type)