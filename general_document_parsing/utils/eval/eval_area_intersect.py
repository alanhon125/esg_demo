import os.path
from collections import namedtuple, Counter
import re
import glob
import json
import numpy as np

pred_dir = '../../data/json_rule/'  # /path/to/doc_parser_result_json
gt_dir = '../../data/gt_bboxes/'  # /path/to/gt_bboxes_json
out_dir = '../../data/text_grp_eval_rule/'  # /path/to/output_eval_result_json

# pred_files = sorted(glob.glob(pred_dir+"*.docparse_json"), key=lambda tup: (tup.split('/')[-1].split('_')[0]))
gt_files = sorted(glob.glob(gt_dir+"*.docparse_json"), key=lambda tup: (tup.split('/')[-1].split('_')[0]))
# files = list(zip(pred_files,gt_files))
# print(files)

for gt_file in gt_files:
    pred_file = os.path.join(pred_dir,re.sub('_gt_bboxes','',os.path.basename(gt_file)))
    fname = os.path.basename(pred_file).split('.')[0]
    print(f'filename: {fname}')
    pred = json.load(open(pred_file))
    gt = json.load(open(gt_file))
    all_gt = list(gt.values())
    page_num = len(gt.values())
    cor_threshold = 2/3
    res = {'filename': fname,
           'overall_performance': {},
           "correct_area_overlap_threshold": round(cor_threshold,2)}
    res.update(dict.fromkeys(range(1, pred['page_num']+1)))
    assert pred["filename"] == list(gt.keys())[0], "Mismatch files with prediction: {} & ground truth: {}".format(pred_file, gt_file)
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    num_pred_bbox = {} # key: page_id, value: count
    for i in pred['content']:
        page_id = i['page_id']
        if page_id not in num_pred_bbox:
            num_pred_bbox[page_id] = 0
        num_pred_bbox[page_id]+=1
    num_gt_bbox = {int(k):len(v) for k,v in gt[fname].items()}

    for block in pred['content']: # For each detected text block
        block_page = int(block['page_id'])
        assert str(block_page) in all_gt[0], "Miss ground truth labeling on {} page_{}".format(os.path.basename(pred_file).split('.')[0],block_page)
        gt_bboxes = all_gt[0][str(block_page)] # get gt_bboxes in page=block_page
        pred_bbox = block['item_bbox']
        pred_rect = Rectangle(*pred_bbox)
        pred_area = (pred_rect.xmax - pred_rect.xmin)*(pred_rect.ymax - pred_rect.ymin)

        iog_list = []
        iod_list = []
        for box in gt_bboxes: # compare with each ground truth bounding box

            gt_rect = Rectangle(*box)
            gt_area = (gt_rect.xmax - gt_rect.xmin)*(gt_rect.ymax - gt_rect.ymin)

            def intersect_area(a, b):  # returns None if rectangles don't intersect
                dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
                dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
                if (dx>=0) and (dy>=0):
                    return dx*dy
                else:
                    return 0

            iog = round(intersect_area(gt_rect, pred_rect)/gt_area,2)
            iod = round(intersect_area(gt_rect, pred_rect)/pred_area,2)
            iog_list.append(iog)
            iod_list.append(iod)

        if not res[block_page]:
            res[block_page] = {}
        if 'iog' not in res[block_page]:
            res[block_page]['iog'] = []
        if 'iod' not in res[block_page]:
            res[block_page]['iod'] = []

        res[block_page]['iog'].append(iog_list)
        res[block_page]['iod'].append(iod_list)

    all_correct = 0
    all_miss = 0
    all_false = 0
    all_split = 0
    all_merge = 0
    all_split_merge = 0

    for i in range(1,pred['page_num']+1):
        if not res[i]:
            continue
        correct = 0
        miss = 0
        false = 0
        split = 0
        merge = 0
        split_merge = 0
        iogs = res[i]['iog']                        # num_pred x num_gt
        iogs_trans = list(map(list, zip(*iogs)))    # num_gt x num_pred
        iods = res[i]['iod']                        # num_pred x num_gt
        iods_trans = list(map(list, zip(*iods)))    # num_gt x num_pred

        iog_iods_trans = list(zip(iogs_trans,iods_trans))
        iog_iods = list(zip(iogs,iods))

        for iog, iod in iog_iods_trans:
            if all((k==0) for k in iog) or all((k==0) for k in iod):
                all_miss += 1
                miss += 1
            elif any((k >= cor_threshold) for k in iog) and any((k >= cor_threshold) for k in iod):
                all_correct += 1
                correct += 1
            elif all((k < cor_threshold) for k in iog) and any((k >= cor_threshold) for k in iod):
                all_split += 1
                split += 1
            elif any((k >= cor_threshold) for k in iog) and all((k < cor_threshold) for k in iod):
                all_merge += 1
                merge += 1
            elif all((k < cor_threshold) for k in iog) and all((k < cor_threshold) for k in iod):
                all_split_merge += 1
                split_merge += 1
        for iog, iod in iog_iods:
            if all((k==0) for k in iog) or all((k==0) for k in iod):
                all_false += 1
                false += 1
        res[i].update({"detection_performance":{'correct': correct, 'miss': miss, 'false': false, 'split': split, 'merge': merge, 'split&merge': split_merge},
                'num_of_blocks':{'ground_truth': num_gt_bbox[i],'dectection': num_pred_bbox[i]}})

    total = all_miss + all_correct + all_split + all_merge + all_false + all_split_merge
    correct_percent = round(all_correct/total,2)
    miss_percent = round(all_miss / total, 2)
    false_percent = round(all_false / total, 2)
    split_percent = round(all_split / total, 2)
    merge_percent = round(all_merge / total, 2)
    split_merge_percent = round(all_split_merge / total, 2)
    precision = round(all_correct / (all_correct + all_false + all_split + all_merge + all_split_merge),2)
    recall = round(all_correct / (all_correct + all_miss),2)
    f1 = round(2* precision * recall / (precision + recall),2)
    res.update({'overall_performance': {'correct_rate': correct_percent, 'miss_rate': miss_percent, 'false_rate': false_percent, 'split_rate': split_percent, 'merge_rate': merge_percent, 'split&merge_rate': split_merge_percent,
                                        'precision': precision, 'recall': recall, 'f1-score': f1
                                        }})

    isExist = os.path.exists(out_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(out_dir)
    with open(os.path.join(out_dir,"{}_txt_grp_eval.docparse_json".format(fname)), 'w') as json_out:
        json_out.write(json.dumps(res, indent=4, ensure_ascii=False))


