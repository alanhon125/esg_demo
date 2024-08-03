import os
from datasets import Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import tqdm
import gc

from general_document_parsing.layoutlm.deprecated.layoutlm.modeling.layoutlm import LayoutlmForTokenClassification
from api.config import *
from postgresql_storage.db_util import DbUtil


def record_status_update(tablename, filter_dict, update_dict):
    du = DbUtil()
    du.update_data(
        tablename,
        filter_dict=filter_dict,
        update_dict=update_dict
    )


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
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    process_id = re.search(r'\d+', progress_name).group()
    du = DbUtil()
    try:
        process_progress = du.select_table(table_name=tablename, field_list=[progress_name], filter_dict=filter_dict)[0][
            progress_name]
    except:
        return
    if not process_progress:
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


def token_classification(data, model_path, fname):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    filename = fname + '.pdf'
    filename_dict = {'filename': filename}
    batch_size = 128
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:2' if use_cuda else 'cpu')
    # device_ids = [2,3]
    dataset = Dataset.from_pandas(data)
    column_names = dataset.column_names
    # features = dataset.features
    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        labels_list = list(unique_labels)
        labels_list.sort()
        return labels_list

    label_column_name = "ner_tags"
    # labels = get_label_list(dataset[label_column_name])
    labels = ['caption', 'figure', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length"
    text_column_name = "tokens"
    img_size = (224, 224)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        use_auth_token=None,
    )

    model = LayoutlmForTokenClassification.from_pretrained(model_path)

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            imagepath = examples["image_path"][org_batch_index]
            image = Image.open(imagepath).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.PILToTensor()
            ])
            image = transform(image)
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        return tokenized_inputs

    input_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_columns,
        num_proc=None,
        load_from_cache_file=None,
        # features=features
    )
    input_dataset.set_format(type="torch", device=device)
    inference_dataloader = DataLoader(input_dataset, batch_size=batch_size)
    # if use_cuda:
    #     model = DataParallel(model, device_ids=device_ids)

    model.to(device)
    # metric = load_metric("seqeval")
    model.eval()
    all_predictions = []
    all_true = []
    t = tqdm.tqdm(inference_dataloader)
    for batch in t:
        record_progress_update(t, PDFFILES_TABLENAME, filename_dict, 'process2_progress',
                               'Document parsing is processing ... ', 0.89)
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            # image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            # predictions
            predictions = outputs[-1].argmax(dim=2)
            # Remove ignored index (special tokens)
            true_predictions = [
                [id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            flat_predictions = [item for sublist in true_predictions for item in sublist]
            flat_true = [item for sublist in true_labels for item in sublist]
            all_predictions.extend(flat_predictions)
            all_true.extend(flat_true)

            # metric.add_batch(predictions=true_predictions, references=true_labels)

            del batch
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_predictions