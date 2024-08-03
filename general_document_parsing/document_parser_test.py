"""
Source: https://towardsdatascience.com/extracting-headers-and-paragraphs-from-pdf-using-pymupdf-676e8421c467
Extracting textual elements (headers, subscripts and paragraphs) from searchable pdf using PyMuPDF
"""

from operator import itemgetter
import fitz
import json
import ndjson
import re
import requests
import pandas as pd
import numpy as np
import argparse
import os
import datetime
import pytz
from statistics import mode
from collections import defaultdict
import unicodedata
import pytesseract
import cv2
import imutils
import spacy
import camelot
import warnings
import sys
import torch

warnings.filterwarnings("ignore")

try:
    from general_document_parsing.layoutlmv2_parse import token_classification
    from general_document_parsing.utils import *
    from api.config import *
except ModuleNotFoundError:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    parentparentdir = os.path.dirname(parentdir)
    sys.path.insert(0, parentdir)
    from utils import *
    from layoutlmv2_parse import token_classification
    sys.path.insert(0, parentparentdir)
    from config import *

def timeit(func):
    from functools import wraps
    import time

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        row = {'task': func.__name__,
               'filename': args[0].fname,
               #    'num_pages': args[0].doc.page_count,
               #    'num_tokens': len(args[0].tokens),
               #    'apply_model': args[0].use_model,
               'runtime': total_time}
        log_task2csv(LOG_DIR + '/log_document_parsing.csv', row)
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def transform_dict_list(tup, args):
    '''
    if current text element is a "list", try to transform into dictionary with key: numbering, value: content, e.g. "(a) XXX, (b) YYY" -> {"(a)": "XXX", "(b)": "YYY"} and return a dictionary
    else if text is other type of text element, try to split the string into phrases with delimiters (at comma or semi-colon or some pattern) and return list of split strings
    '''

    block_id, txt_elem, text = tup
    document_type, nlp, override_key, txt_blocks = args
    cond_split_at_comma = document_type == 'termSheet' and \
        "section" in txt_blocks['content'][block_id] and \
        re.match('|'.join(['.*'+i for i in string_with_whitespace(
            SECTION_SPLIT_AT_COMMA)]), txt_blocks['content'][block_id]["section"], re.IGNORECASE)

    if cond_split_at_comma:
        delimiter_pattern = COMMA_SPLIT_PATTERN
    else:
        delimiter_pattern = PARA_SPLIT_PATTERN

    if isinstance(text, str):
        if document_type == 'termSheet' or (document_type == 'agreement' and 'list' in txt_elem):
            transformed_text = list2dict_2(
                text, nlp, use_nested=False, override_key=override_key, delimiter_pattern=delimiter_pattern)
            if isinstance(transformed_text, dict) and document_type == 'termSheet':
                transformed_text = dict_value_phrase_tokenize(
                    transformed_text, nlp, delimiter_pattern=delimiter_pattern)

    if (isinstance(text, list) or isinstance(text, tuple)) and cond_split_at_comma:
        tmp_text = []
        for txt in text:
            tup = (block_id, txt_elem, txt)
            _, _, c = transform_dict_list(tup, args)
            if isinstance(c, str):
                c = [c]
            tmp_text.extend(c)
        transformed_text = tmp_text

    if 'transformed_text' in locals():
        return (block_id, txt_elem, transformed_text)
    else:
        return (block_id, txt_elem, text)


def batch_transform_dict_list(lst, args):
    results = []
    for tup in lst:
        block_id, txt_elem, text = transform_dict_list(tup, args)
        results.append((block_id, txt_elem, text))
    return results


class DocParser(object):
    """
    @param pdf_inpath: path to pdf
    @type pdf_inpath: str
    @param json_outdir: the output directory where the output json will be written
    @type json_outdir: str
    @param img_outdir: the output directory where the output page images will be written
    @type img_outdir: str
    @param use_model: boolean parameter that indicate to use model for token classification
    @type use_model: bool
    @param do_annot: boolean parameter that indicate to annotate pdf for visualization of results
    @type do_annot: bool
    @param model_path: the input directory where the model stored
    @type model_path: str
    @param annot_outdir: the output directory where the annotated pdf will be written (default: 'annot_pdf/')
    @type model_path: str
    """

    def __init__(self, pdf_inpath, json_outdir, img_outdir, use_model, use_ocr, model_path, annot_outdir, do_annot=True, document_type='esgReport', dev_mode=DEV_MODE):
        self.dev_mode = dev_mode
        self.pdf_inpath = pdf_inpath
        self.json_outdir = json_outdir
        self.img_outdir = img_outdir
        self.annot_outdir = annot_outdir
        self.do_annot = do_annot
        self.document_type = document_type
        self.use_model = use_model
        self.use_ocr = use_ocr
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.model_input_data = []
        self.fname = os.path.basename(os.path.splitext(pdf_inpath)[0])
        self.doc = fitz.open(self.pdf_inpath)
        self.annot_order = fitz.open(self.pdf_inpath)
        self.annot_ele = fitz.open(self.pdf_inpath)
        self.annot_model_ele = fitz.open(self.pdf_inpath)
        self.annot_txtblock = fitz.open(self.pdf_inpath)
        self.page_num = self.doc.page_count
        self.page_id2size = defaultdict()
        self.styles = {}
        self.font_counts = {}
        self.font_property_tag = {}
        self.doc_tag_set = {}
        self.num_head = None
        self.num_sub = None
        self.txt_ele = []  # list of token tag dictionaries
        self.txt_model_ele = []  # list of token model-inference tag dictionaries
        self.txt_blocks = {}
        self.tokens = []
        self.text = ''
        self.pageid_bbox2word = {}
        if document_type == 'esgReport':
            self.line_spacing_split_tags = ['title', 'section', 'caption']
        else:
            self.line_spacing_split_tags = ['section', 'paragraph']
        self.ner_tags = ['caption', 'figure', 'footer', 'list',
                         'paragraph', 'reference', 'section', 'table', 'title']
        self.annot_style_color = {'abstract': (1, 0, 1), 'author': (0.5, 0, 0.5), 'caption': (0, 1, 0),
                                  'date': (1, 1, 1), 'equation': (0.5, 0.5, 0.5), 'figure': (1, 0.85, 0),
                                  'footer': (0, 0, 1), 'list': (0, 0.75, 1), 'paragraph': (1, 0, 0),
                                  'reference': (0.5, 0.5, 0),
                                  'section': (0.2, 0.8, 0.2), 'table': (1, 0.65, 0), 'title': (0, 0.5, 0)}
        self.footer_lower_tolerance = 80
        self.footer_upper_tolerance = 1000 - 80
        self.table = None
        self.mode_line_spacing = {}
        self.max_line_spacing = {}
        if document_type == 'termSheet':
            self.para_upper_quartile_line_spacing = 15
            self.section_upper_quartile_line_spacing = 15
        else:
            self.para_upper_quartile_line_spacing = 10
            self.section_upper_quartile_line_spacing = 5
        self.mode_word_spacing = {}
        self.max_word_spacing = {}
        self.min_word_spacing = {}
        self.override_key = None
        self.last_table_header = None
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            os.system('python3 -m spacy download en_core_web_md')
            self.nlp = spacy.load("en_core_web_md")
        self.nlp = config_nlp_model(self.nlp)
        self.process_font_tag()

    @timeit
    def fonts_line_spacing(self):
        """Extracts fonts, max. line spacing and their usage in PDF documents.
        @rtype: [(font_size, count), (font_size, count)], dict
        @return: most used fonts sorted by count, font style information
        """

        for page in self.doc:
            normalized_line_spacings = []
            w, h = page.rect.width, page.rect.height
            page_id = page.number + 1
            blocks = page.get_text("dict")["blocks"]
            blocks = sorted(blocks, key=lambda x: (
                x['bbox'][1], x['bbox'][0], x['bbox'][3], x['bbox'][2]))
            prev_span_bbox = None
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # block contains text
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text span
                            identifier = "{0}_{1}_{2}_{3}".format(
                                s['size'], s['flags'], s['font'], s['color'])
                            normalized_bbox = normalize_bbox(s['bbox'], w, h)
                            if prev_span_bbox:
                                xoffset, yoffset = min_offset(
                                    normalized_bbox, prev_span_bbox)
                                line_spacing = yoffset
                                normalized_line_spacings.append(line_spacing)
                            prev_span_bbox = normalized_bbox
                            self.styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                       'color': s['color']}
                            self.font_counts[identifier] = self.font_counts.get(
                                identifier, 0) + 1  # count the fonts usage
            if len(normalized_line_spacings) > 1:
                self.mode_line_spacing[page_id] = mode(
                    normalized_line_spacings)
                self.max_line_spacing[page_id] = max(normalized_line_spacings)
            else:
                self.mode_line_spacing[page_id] = 0
                self.max_line_spacing[page_id] = 0
                # self.upper_quartile_line_spacing[page_id] = quantiles(normalized_line_spacings, n=100)[94]
        self.font_counts = sorted(
            self.font_counts.items(), key=itemgetter(1), reverse=True)
        if len(self.font_counts) < 1:
            raise ValueError(
                f"Zero discriminating fonts found for document {self.fname}!")

    @timeit
    def word_spacing(self):

        for page in self.doc:
            normalized_word_spacings = []
            page_id = page.number + 1
            width, height = page.rect.width, page.rect.height
            words = page.get_text("words")
            prev_word_bbox = None
            for w in words:
                x0, y0, x1, y1, word, blocknumber, linenumber, wordnumber = w
                self.pageid_bbox2word.update({(page_id, x0, y0, x1, y1): word})
                bbox = (x0, y0, x1, y1)
                normalized_bbox = normalize_bbox(bbox, width, height)
                if prev_word_bbox:
                    xoffset, yoffset = min_offset(
                        normalized_bbox, prev_word_bbox)
                    if yoffset <= self.max_line_spacing[page_id]:
                        word_spacing = xoffset
                        normalized_word_spacings.append(word_spacing)

                prev_word_bbox = normalized_bbox
            if len(normalized_word_spacings) > 1:
                self.mode_word_spacing[page_id] = mode(
                    normalized_word_spacings)
                self.max_word_spacing[page_id] = max(normalized_word_spacings)
                self.min_word_spacing[page_id] = min(normalized_word_spacings)
            else:
                self.mode_word_spacing[page_id] = 0
                self.max_word_spacing[page_id] = 0
                self.min_word_spacing[page_id] = 0

    @timeit
    def extract_text_by_bbox(self, page_id, bbox):
        '''
        Given page id (starting from 1), actual target query bounding box and pageid_bbox2word dictionary with key-value pairs (page_id, x0, y0, x1, y1): word
        return a string that bound by this bounding box
        @param page_id: query document page id (starting from 1)
        @type page_id: int
        @param bbox: actual target query bounding box in [x0,y0,x1,y1]
        @type bbox: list or tuple
        '''
        import ast
        string = ''
        tx0, ty0, tx1, ty1 = bbox
        for key, word in self.pageid_bbox2word.items():
            if isinstance(key, str):
                key = ast.literal_eval(key)
            pid, x0, y0, x1, y1 = key
            if pid == page_id and x0 >= tx0 and y0 >= ty0 and x1 <= tx1 and y1 <= ty1:
                string += word + ' '
        return string.strip()

    @timeit
    def font_tags(self):
        """Returns dictionary with font sizes as keys and tags as value.
        @rtype: dict
        @return: all element tags based on font-sizes
        """
        '''
        source: https://pymupdf.readthedocs.io/en/latest/recipes-text.html?highlight=flags#how-to-analyze-font-characteristics
        flag represents font properties except for the first bit 0. They are to be interpreted like this: 
        {
            0: 'Regular|Normal',
            2**0: 'Superscripted', # flags = 1
            2**1: 'Italic', # flags = 2
            2**2: 'Serifed', # flags = 4
            2**3: 'Monospaced', # flags = 8
            2**4: 'Bold', # flags = 16
        }
        '''
        p_style = self.styles[self.font_counts[0][0]
                              ]  # get style for most used font by count (paragraph)
        p_size = p_style['size']  # get the paragraph's size
        p_color = p_style['color']  # get the paragraph's color
        p_font = p_style['font']  # get the paragraph's font
        p_flags = p_style['flags']  # get the paragraph's flag

        font_sizes_color = []
        for (identifier, count) in self.font_counts:
            font_tup = identifier.split('_')
            # if tuple is not length 4, i.e. not (font_size, font_flag, font_name, font_color)
            if len(font_tup) != 4:
                font_tup[2] = '_'.join(font_tup[2:-1])
                for i in font_tup[3:-1]:
                    font_tup.remove(i)
            font_size = float(font_tup[0])
            font_flag = int(font_tup[1])
            font = font_tup[2]
            font_color = int(font_tup[-1])
            if font_size == p_size and font == p_font and font_color == p_color and font_flag == p_flags:
                font_style = -1
            else:
                font_style = font_flag
            font_sizes_color.append((font_size, font, font_color, font_style))
        font_sizes_color = sorted(
            font_sizes_color, key=lambda tup: (tup[0], tup[-1]), reverse=True)
        # aggregating the tags for each font size
        idx = tmp_idx = 0
        for seq, (size, font, color, font_style) in enumerate(font_sizes_color):
            same_p_size = size == p_size
            same_p_color = color == p_color
            bold = font_style >= 2 ** 4
            if seq > 0:  # if previous size and font are the same but different color, cancel the update heading/subscript counter
                prev_size = font_sizes_color[seq - 1][0]
                prev_font = font_sizes_color[seq - 1][1]
                prev_tag = list(self.font_property_tag.values())[-1]
                if prev_size == size and prev_font == font:
                    idx -= 1
            if (same_p_size and ((not same_p_color) or (same_p_color and bold))) or size > p_size:
                if seq > 0 and prev_tag == 'paragraph':
                    idx = tmp_idx
                self.font_property_tag[(
                    size, font, color, font_style)] = 'heading{0}'.format(idx)
                idx += 1
                tmp_idx = idx
            elif same_p_size and (same_p_color and not bold):
                idx = 0
                self.font_property_tag[(
                    size, font, color, font_style)] = 'paragraph'
            elif size < p_size:
                self.font_property_tag[(
                    size, font, color, font_style)] = 'subscript{0}'.format(idx)
                idx += 1
        self.num_head = len(
            set([e for e in self.font_property_tag.values() if re.search('heading', e)]))
        self.num_sub = len(
            set([e for e in self.font_property_tag.values() if re.search('subscript', e)]))

    def style2tag(self, style):
        """
        Map fine-grained font style (heading_n, subscript_n, paragraph) to general element tag (title, section, caption, paragraph, list, footer)
        @param style: font style expressed in heading_n, subscript_n, paragraph
        @type style: str
        @rtype: str
        @return: general element tag (title, section, caption, paragraph, list, footer)
        """
        if re.search('heading|subscript', style):
            style_id = int(re.findall(r'\d+', style)[0])
        if re.search('heading', style):
            if self.num_head < 20:
                if style_id <= self.num_head / 2:
                    return 'title'
                elif self.num_head / 2 < style_id <= self.num_head * 2 / 3:
                    return 'section'
                else:
                    return 'caption'
            else:
                if style_id <= self.num_head * 0.3:
                    return 'title'
                elif self.num_head * 0.3 < style_id <= self.num_head * 0.4:
                    return 'section'
                elif self.num_head * 0.4 < style_id <= self.num_head * 0.6:
                    return 'caption'
                else:
                    return 'paragraph'
        elif re.search('subscript', style):
            if self.num_sub < 20:
                if style_id <= self.num_sub / 2:
                    return 'list'
                else:
                    return 'footer'
            else:
                if style_id <= self.num_sub * 0.6:
                    return 'paragraph'
                elif self.num_sub * 0.6 < style_id <= self.num_sub * 0.8:
                    return 'list'
                else:
                    return 'footer'
        else:
            return 'paragraph'

    @timeit
    def elements_grouping(self, txt_ele):
        '''
        Group token_rule_tag or tokens with same style
        @return: list of grouped element represents in [tag, s, string_len, page_id, element_cluster_outline_bbox]
        @rtype: list
        '''
        if txt_ele:
            bbox_cluster = []
            output = []
            iter_txt_ele = neighborhood(txt_ele)
            isFirst = True
            for prev, curr, nxt in iter_txt_ele:
                tag = curr['tag']
                if tag in ['footer', 'reference']:
                    continue
                if 'token' in curr:
                    string = curr['token']
                elif 'text' in curr:
                    string = curr['text']
                if tag in ['section'] and string == ':':
                    continue
                page_id = curr['page_id']
                bbox = curr['bbox']
                if isFirst:  # if first
                    s = string
                    bbox_cluster.append(bbox)
                    isFirst = False
                    last_tag = tag
                    last_pageid = page_id
                    continue
                else:
                    prev_tag = prev['tag']
                    prev_pageid = prev['page_id']
                    prev_bbox = prev['bbox']
                element_cluster_outline_bbox = bboxes_cluster_outline(
                    bbox_cluster)
                if tag == 'section':
                    is_adjacent = is_adjacent_bbox(bbox, prev_bbox, xoffset_thres=1000, yoffset_thres=self.section_upper_quartile_line_spacing)
                elif tag == 'paragraph':
                    is_adjacent = is_adjacent_bbox(bbox, prev_bbox, xoffset_thres=1000, yoffset_thres=self.para_upper_quartile_line_spacing)
                else:
                    is_adjacent = is_adjacent_bbox(bbox, prev_bbox, xoffset_thres=1000, yoffset_thres=self.max_line_spacing[page_id])
                grp_token_info = {
                    'text': s,
                    'text_length': len(s),
                    'token_bboxes': bbox_cluster,
                    'bbox': element_cluster_outline_bbox
                }
                if (tag == prev_tag and page_id == prev_pageid and is_adjacent) or (tag == prev_tag and page_id != prev_pageid):
                    s += ' ' + string
                else:
                    grp_token_info['tag'] = last_tag
                    grp_token_info['page_id'] = last_pageid
                    output.append(grp_token_info)
                    bbox_cluster = []
                    s = string
                bbox_cluster.append(bbox)
                last_tag = tag
                last_pageid = page_id

                # if last
                if not nxt or (nxt['tag'] == 'footer' and txt_ele.index(nxt) == len(txt_ele) - 1):
                    grp_token_info = {
                        'text': s,
                        'text_length': len(s),
                        'token_bboxes': bbox_cluster,
                        'bbox': bboxes_cluster_outline(bbox_cluster)
                    }
                    grp_token_info['tag'] = tag
                    grp_token_info['page_id'] = page_id
                    output.append(grp_token_info)
            return output
        else:
            raise ValueError(
                "The text elements have not been classified. Please run DocParser.layoutlm_token_classify()")

    @timeit
    def extract_token2(self):
        for page in self.doc:
            page_id = page.number + 1
            w, h = page_size(self.doc, page_id - 1)
            self.page_id2size[int(page_id)] = {'width': w, 'height': h}

        input_data_path = OUTPUT_LAYOUTLM_INPUT_DIR + self.fname + '.json'
        if (not os.path.exists(input_data_path) and self.dev_mode) or not self.dev_mode:
            for page in self.doc:
                page_id = page.number + 1
                img_path = os.path.join(
                    self.img_outdir, self.fname + '_{}_ori.jpg'.format(str(page_id)))
                model_input = {
                    'id': int(page_id),
                    'tokens': [],
                    'bboxes': [],
                    'ner_tags': [],
                    'image_path': img_path
                }
                width, height = page.rect.width, page.rect.height
                words = page.get_text("words")
                if self.document_type == 'agreement':
                    # sorting block from top to bottom then from left to right
                    words = sorted(words, key=lambda w: (
                        w[1], w[0], w[3], w[2]))
                prev_word_bbox = None
                for i, w in enumerate(words):
                    x0, y0, x1, y1, word, blocknumber, linenumber, wordnumber = w
                    bbox = (x0, y0, x1, y1)
                    normalized_bbox = normalize_bbox(bbox, width, height)
                    model_input['tokens'].append(word)
                    model_input['bboxes'].append(normalized_bbox)
                    model_input['ner_tags'].append('paragraph')
                self.model_input_data.append(model_input)

            if self.dev_mode:
                if not os.path.exists(OUTPUT_LAYOUTLM_INPUT_DIR):
                    os.makedirs(OUTPUT_LAYOUTLM_INPUT_DIR)
                with open(input_data_path, 'w') as output:
                    ndjson.dump(self.model_input_data,
                                output, ensure_ascii=False)
        else:
            with open(input_data_path, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    self.model_input_data.append(d)

    @timeit
    def extract_token(self):
        """
        By using PyMUPDF, iterate text blocks and lines to extract tokens, bounding boxes and element tagging in each searchable PDF page
        store a list of token,bboxes,tag info dictionaries
        """
        token_rule_tag = []  # list with headers and paragraphs
        bbox_cluster = []
        first = True  # boolean operator for first header
        isPrevChinese = False
        fullwidth_regex = re.compile(FULLWIDTH_ASCII_VARIANTS)

        input_data_path = OUTPUT_LAYOUTLM_INPUT_DIR + self.fname + '.json'
        for page in self.doc:
            page_id = page.number + 1
            w, h = page_size(self.doc, page_id - 1)
            self.page_id2size[int(page_id)] = {'width': w, 'height': h}

        if (not os.path.exists(input_data_path) and self.dev_mode) or not self.dev_mode:
            for page in self.doc:
                page_id = page.number + 1
                # create dictionary of tokens, bbox, ner_tags and image in each page for model detection of token classes
                img_path = os.path.join(
                    self.img_outdir, self.fname + '_{}_ori.jpg'.format(str(page_id)))
                model_input = {
                    'id': int(page_id),
                    'tokens': [],
                    'bboxes': [],
                    'ner_tags': [],
                    'image_path': img_path
                }
                blocks = page.get_text("dict")['blocks']
                if self.document_type == 'agreement':
                    blocks = sorted(blocks, key=lambda b: (b['bbox'][1],
                                                           b['bbox'][0],
                                                           b['bbox'][3],
                                                           b['bbox'][2]))  # sorting block from top to bottom then from left to right
                # blocks = [i for i in page.get_text("dict")["blocks"] if not any((k<0) for k in i['bbox'])] # ignore block if any x,y coordinate beyond page boundaries

                # only iterate pages that contain text
                if any((b['type'] == 0) for b in blocks):
                    for b in blocks:  # iterate through the text blocks
                        if b['type'] == 1 and self.use_ocr:  # this block contains image
                            bbox = b['bbox']
                            # e.g.  b'\xff\xd8\xff\xe0\x00\x10...'
                            image_bytes = b['image']
                            image_np = np.frombuffer(image_bytes, np.uint8)
                            image = cv2.imdecode(image_np, -1)
                            image = remove_black_background(image)
                            image = white_pixels_to_transparent(image)
                            # reference: https://stackoverflow.com/questions/58103337/how-to-ocr-image-with-tesseract
                            image = imutils.resize(image, width=400)
                            # Gaussian blur
                            blur = cv2.GaussianBlur(image, (7, 7), 0)
                            splits = cv2.split(blur)
                            # Otsu's threshold
                            streams = [cv2.threshold(
                                stream, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for stream in splits]
                            # Invert image
                            newRGBImage = cv2.merge(streams)
                            # show image
                            # cv2.imshow('Image', newRGBImage)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()  # It destroys the image showing the window.
                            # give the numpy array directly to pytesseract, no PIL or other acrobatics necessary
                            token = pytesseract.image_to_string(
                                newRGBImage, lang="eng", config='--psm 6 -c tessedit_char_whitelist=1234567890\ \&\(\)\.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
                            if token:
                                token = token.strip()
                                s_tag = 'section'
                                w = self.page_id2size[int(page_id)]['width']
                                h = self.page_id2size[int(page_id)]['height']
                                bbox = normalize_bbox(bbox, w, h)
                                x0, y0, x1, y1 = bbox
                                # if any x,y coordinate beyond page boundaries, discard the text
                                if any((c <= 0) or (c >= 1000) for c in bbox):
                                    continue
                                if page_id != 1 and (y1 <= self.footer_lower_tolerance or y1 >= self.footer_upper_tolerance):
                                    s_tag = 'footer'
                                if token == 'n':  # sometimes bullet-point character recognize as 'n', replace it with geometric shape
                                    token = '•'
                                model_input['tokens'].append(token)
                                model_input['bboxes'].append(bbox)
                                model_input['ner_tags'].append(s_tag)
                                self.tokens.append({
                                    'token': token,
                                    'bbox': bbox,
                                    'page_id': page_id,
                                    'rule_tag': s_tag
                                })
                                token_rule_tag.append({
                                    'token': token,
                                    'bbox': bbox,
                                    'page_id': page_id,
                                    'tag': s_tag
                                })
                                bbox_cluster.append(bbox)
                        elif b['type'] == 0:  # this block contains text
                            # REMEMBER: multiple fonts and sizes are possible IN one block
                            block_string = ""  # text found in block
                            tmp_token = ''
                            # iterate through the text lines
                            for line_no, l in enumerate(b["lines"]):
                                # iterate through the text span
                                for span_no, s in enumerate(l["spans"]):
                                    token = s['text'].strip()
                                    bbox = s['bbox']
                                    if token:
                                        # The “NFKC” stands for “Normalization Form KC [Compatibility Decomposition, followed by Canonical Composition]”, and replaces full-width characters by half-width ones
                                        token = unicodedata.normalize(
                                            "NFKD", token)
                                     # discard string with unwanted characters
                                    if self.use_model and token:
                                        # LayoutLMv3 model doesn't accept Chinese characters, so we need to remove Chinese character and any character in-between Chinese characters
                                        isCurrTokenFullWidth = fullwidth_regex.search(
                                            token)
                                        if span_no + 1 <= len(l["spans"]) - 1:
                                            next_token = l["spans"][span_no + 1]['text']
                                            isNextTokenFullWidth = fullwidth_regex.search(
                                                next_token)
                                            next_token = discard_str_with_unwant_char(
                                                next_token)
                                            if ((next_token is None or isNextTokenFullWidth) and isPrevChinese) or (isCurrTokenFullWidth and isNextTokenFullWidth):
                                                token = None
                                        if token is not None:
                                            token = discard_str_with_unwant_char(
                                                token)
                                        if token is None:
                                            isPrevChinese = True
                                        else:
                                            isPrevChinese = False
                                    if self.use_model and token:
                                        token = filter_invalid_char(token)
                                    if token:  # if token is not ''
                                        # Handle surrogates characters that do not have a valid representation in Unicode
                                        try:
                                            token = token.encode(
                                                'utf-8', 'ignore').decode('utf-8')
                                        except UnicodeEncodeError:
                                            continue
                                        token = token.expandtabs(1)
                                        if token == 'n':  # sometimes bullet-point character recognize as 'n', replace it with geometric shape
                                            token = '•'
                                        # add whitespace as delimiter to seperate camel case, e.g. theBorrower -> the Borrower
                                        token = add_space2camel_case(token)
                                        identifier = "{0}_{1}_{2}_{3}".format(
                                            s['size'], s['flags'], s['font'], s['color'])
                                        if identifier == self.font_counts[0][0]:
                                            font_style = -1
                                        else:
                                            font_style = s['flags']
                                        tag = self.font_property_tag[(
                                            s['size'], s['font'], s['color'], font_style)]
                                        # rename tag from headingn to title/section/caption and from subscriptn to list/footer
                                        s_tag = self.style2tag(tag)
                                        x0, y0, x1, y1 = bbox
                                        bbox = normalize_bbox(bbox, self.page_id2size[int(
                                            page_id)]['width'], self.page_id2size[int(page_id)]['height'])
                                        _, _, _, ny1 = bbox
                                        # if any x,y coordinate beyond page boundaries, discard the text
                                        if any((c <= 0) or (c >= 1000) for c in bbox):
                                            continue
                                        if page_id != 1 and (ny1 <= self.footer_lower_tolerance or ny1 >= self.footer_upper_tolerance):
                                            s_tag = 'footer'
                                        # previous text and current text very close to each other
                                        if not first and abs(x0-px1) <= 0.015 and abs(y0-py0) <= 2 and span_no == previous_span_no + 1:
                                            previous_token = model_input['tokens'].pop(
                                                -1)
                                            model_input['bboxes'].pop(-1)
                                            model_input['ner_tags'].pop(-1)
                                            self.tokens.pop(-1)
                                            bbox_cluster.pop(-1)
                                            token = previous_token.strip() + ' ' + token.strip()
                                            bbox = normalize_bbox([px0, py0, x1, y1], self.page_id2size[int(
                                                page_id)]['width'], self.page_id2size[int(page_id)]['height'])
                                        token = filter_invalid_char(token)
                                        model_input['tokens'].append(token)
                                        model_input['bboxes'].append(bbox)
                                        model_input['ner_tags'].append(s_tag)
                                        self.tokens.append({
                                            'token': token,
                                            'bbox': bbox,
                                            'page_id': page_id,
                                            'rule_tag': s_tag,
                                        })
                                        bbox_cluster.append(bbox)
                                        if first:
                                            first = False
                                            block_string = s['text']
                                        else:
                                            if s_tag == previous_s_tag:
                                                if block_string and all((c == "\n") for c in block_string):
                                                    # block_string only contains pipes
                                                    block_string = ''
                                                    block_string += s['text']
                                                if block_string == "":
                                                    # new block has started, so append size tag
                                                    block_string = s['text']
                                                else:  # in the same block, so concatenate strings
                                                    block_string += " " + \
                                                        s['text']
                                            else:
                                                if block_string != '' and not all((c == "\n") for c in block_string) and not (abs(x0-px1) <= 0.015 and abs(y0-py0) <= 2 and span_no == previous_span_no + 1):
                                                    bbox_cluster.pop(-1)
                                                    element_cluster_outline_bbox = bboxes_cluster_outline(
                                                        bbox_cluster)
                                                    # count the length of string ignore tag and linebreak
                                                    token_rule_tag.append({
                                                        'token': block_string,
                                                        'bbox': element_cluster_outline_bbox,
                                                        'page_id': page_id,
                                                        'tag': previous_s_tag,
                                                    })
                                                    bbox_cluster = []
                                                    bbox_cluster.append(bbox)
                                                block_string = s['text']
                                        previous_s = s
                                        px0, py0, px1, py1 = (x0, y0, x1, y1)
                                        previous_span_no = span_no
                                        previous_s_tag = self.font_property_tag[(
                                            previous_s['size'], previous_s['font'], previous_s['color'], font_style)]
                                        previous_s_tag = self.style2tag(
                                            previous_s_tag)
                                # new block started, indicating with a newline character
                                block_string += "\n"
                            if block_string != '' and not all((c == "\n") for c in block_string):
                                element_cluster_outline_bbox = bboxes_cluster_outline(
                                    bbox_cluster)
                                # count the length of string ignore tag and linebreak
                                token_rule_tag.append({
                                    'token': block_string,
                                    'bbox': element_cluster_outline_bbox,
                                    'page_id': page_id,
                                    'tag': s_tag
                                })
                                bbox_cluster = []
                    self.model_input_data.append(model_input)
            self.txt_ele = token_rule_tag

            if self.dev_mode and not os.path.exists(OUTPUT_LAYOUTLM_INPUT_DIR):
                os.makedirs(OUTPUT_LAYOUTLM_INPUT_DIR)
            if self.dev_mode:
                with open(input_data_path, 'w') as output:
                    ndjson.dump(self.model_input_data,
                                output, ensure_ascii=False)
        else:
            token_rule_tag = []
            with open(input_data_path, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    self.model_input_data.append(d)
                    tokens = d["tokens"]
                    bboxes = d["bboxes"]
                    page_id = [d["id"]] * len(d["tokens"])
                    ner_tags = d["ner_tags"]
                    keys = ['token', 'bbox', 'page_id', 'tag']
                    pairs = list(zip(tokens, bboxes, page_id, ner_tags))
                    dic = [dict(zip(keys, v)) for v in pairs]
                    token_rule_tag.extend(dic)
            self.txt_ele = token_rule_tag
        self.doc_tag_set = set([e['tag'] for e in self.txt_ele])

    @timeit
    def layoutlm_token_classify(self):
        '''
        token classification by layoutlm
        rectify the predictions with some rules and store the predictions
        '''
        output_data_path = OUTPUT_LAYOUTLM_OUTPUT_DIR + self.fname + '.json'
        if self.dev_mode and os.path.exists(output_data_path):
            with open(output_data_path, 'r') as f:
                d = json.load(f)
            # all_tokens, all_bboxes, all_ids, all_true, all_predictions = d["tokens"], d["bboxes"], d["id"], d["ner_tags"], d["predictions"]
            # result = list(zip(all_tokens, all_bboxes, all_ids, all_true, all_predictions))
            # keys = ['token', 'bbox', 'page_id', 'rule_tag', 'tag']
            # self.tokens = [dict(zip(keys, i)) for i in result]
            self.tokens = d
        else:
            # LayoutLM model prediction for token classes
            df = pd.DataFrame(self.model_input_data)
            if torch.cuda.is_available():
                self.tokens = token_classification(df, self.model_path, self.fname, gpu_ids=DOCPARSE_GPU_ID,
                                                   batch_size=LAYOUTLM_BATCH_SIZE, document_type=self.document_type)
            else:
                headers = {'Content-type': 'application/json',
                           'Accept': 'application/json'}

                data = df.to_dict('records')
                # Request layoutlm token classification from GPU server
                data2 = []
                for d in data:
                    d['image_path'] = d['image_path'].replace(
                        PROJ_ROOT_DIR, REMOTE_PROJ_DIR)
                    data2.append(d)
                request = {
                    "data": data2,
                    "document_type": self.document_type
                }
                print('Request layoutlmv3 prediction, waiting for response ...')
                r = requests.post(LAYOUTLMV3_PREDICT_URL,
                                  data=json.dumps(request), headers=headers)
                r = json.loads(r.content.decode('utf-8'))
                self.tokens = r['results']
                print('Complete layoutlmv3 prediction.')

            if self.dev_mode:
                if not os.path.exists(OUTPUT_LAYOUTLM_OUTPUT_DIR):
                    os.makedirs(OUTPUT_LAYOUTLM_OUTPUT_DIR)
                with open(output_data_path, 'w') as output:
                    json.dump(self.tokens, output,
                              indent=4, ensure_ascii=False)

        self.txt_ele = self.tokens
        self.doc_tag_set = set([e['tag'] for e in self.txt_ele])
        # self.txt_model_ele = self.elements_grouping(model_result)

    @timeit
    def token_level_grouping(self):
        '''
        1. Group token_rule_tag or tokens with same style
        2. Then sort the token tag dictionaries from top to bottom, then left to right
        3. Annotate reading order with element tag and bounding boxes info in each PDF page
        '''
        for _ in range(2):
            # first iteration grouping token in token-level that is adjacent and share same text element
            # second iteration grouping text in textbox-level that is adjacent and share same text element
            self.txt_ele = self.elements_grouping(self.txt_ele)

        # self.txt_ele = sorted(self.txt_ele, key=lambda x: (x['page_id'], x['bbox'][1],
        #                                                    x['bbox'][0],
        #                                                    x['bbox'][3],
        #                                                    x['bbox'][2]))

        page_reading_order = 0
        for prev, curr, nxt in neighborhood(self.txt_ele):
            curr_pageid = curr['page_id']
            curr_tag = curr['tag']
            curr_bbox = curr['bbox']
            if prev:
                prev_pageid = prev['page_id']
            if prev and curr_pageid != prev_pageid:
                page_reading_order = 0
            if self.do_annot:
                self.annot_order = annot_pdf_page(
                    self.annot_order,
                    int(curr_pageid) - 1,
                    str(page_reading_order) + ' ' +
                    curr_tag + ' ' + str(curr_bbox),
                    curr_bbox,
                    color=self.annot_style_color[curr_tag])
            page_reading_order += 1

    @timeit
    def txt_block_grp_seg(self):
        """
        Group and segment text blocks with leading font style tags (headings, paragraph and subscripts) and give JSON document of parsed pdf
        """
        item_id = 0
        tmp_bbox = []
        filesize_MB = os.path.getsize(self.pdf_inpath) / (1024 * 1024)

        self.txt_blocks = {
            'filename': self.fname,
            'report_type': self.document_type,
            'page_num': self.txt_ele[-1]['page_id'],
            'pages_size': self.page_id2size,
            'file_size': str(filesize_MB) + ' MB',
            'items_num': 0,
            'process_datetime': None,
            'content': []
        }

        # dictionary to decide whether the current text element append to the previous text block
        # The key is the previous element tag, and
        # the list value means current tag should be associated
        inclusive_with_prev = {
            'caption': ['figure', 'list', 'table', 'paragraph'],
            'figure': ['caption', 'paragraph', 'reference'],
            'footer': [],
            'list': ['caption', 'paragraph'],
            'paragraph': ['caption', 'list', 'paragraph', 'reference', 'table'],
            'reference': [],
            'section': ['caption', 'figure', 'list', 'paragraph', 'table', 'sub_section'],
            'sub_section': ['caption', 'figure', 'list', 'paragraph', 'table', 'sub_sub_section'],
            'sub_sub_section': ['caption', 'figure', 'list', 'paragraph', 'table'],
            'table': ['paragraph', 'reference'],
            'title': ['figure', 'list', 'table']
        }

        pos_lookbehind_list = ['\d{1} \%', '\d{2} \%', '\d{1}\%', '\d{2}\%',
                               '\d{1} percent', '\d{2} percent', '\d{1} per cent', '\d{2} per cent']
        backslash_char = "\\"
        pos_lookbehind = rf'(?!{")(?!".join(pos_lookbehind_list+[rf"{backslash_char}."+i for i in pos_lookbehind_list])})'
        sub_section_pattern = rf'^\d+[a-zA-Z]*\.{pos_lookbehind}\d+\.*\s*.*'
        subsub_section_pattern = r'^\d+[a-zA-Z]*\.\d+\.\d+\.*\s*.*'

        def section_rename(text):
            if re.match(subsub_section_pattern, text):
                return 'sub_sub_section'
            elif re.match(sub_section_pattern, text):
                return 'sub_section'
            else:
                return 'section'

        self.ner_tags.extend(['sub_section', 'sub_sub_section'])

        # Iterating over 3 consecutive (previous, current, next) to compare the style and page id
        iter_grp_txt = neighborhood(self.txt_ele)
        isPageBreakOnSameStyleText = False
        page_id_range = []
        item_bboxes = []
        ele_bboxes = {}

        for prev, curr, nxt in iter_grp_txt:
            if prev:
                prev_style = prev['tag']
                prev_pgid = prev['page_id']
                prev_txt = replace_char(prev['text'])
                prev_bbox = prev['bbox']
                ori_prev_bbox = denormalize_bbox(
                    prev_bbox, self.page_id2size[1]['width'], self.page_id2size[1]['height'])
                if prev_style == 'section':
                    prev_style = section_rename(prev_txt)

            curr_style = curr['tag']
            curr_txt = replace_char(curr['text'])
            curr_pgid = curr['page_id']
            curr_bbox = curr['bbox']
            ori_curr_bbox = denormalize_bbox(
                curr_bbox, self.page_id2size[1]['width'], self.page_id2size[1]['height'])

            if 'table' in curr_style:
                x0, y0, x1, y1 = ori_curr_bbox
                flip_ori_curr_bbox = (x0, abs(
                    y0 - self.page_id2size[1]['height']), x1, abs(y1 - self.page_id2size[1]['height']))
                '''
                table_areas accepts strings of the form x1,y1,x2,y2 
                where (x1, y1) -> top-left and (x2, y2) -> bottom-right in PDF coordinate space. 
                In PDF coordinate space, the bottom-left corner of the page is the origin, with coordinates (0, 0).
                '''
                try:
                    tables = camelot.read_pdf(
                        filepath=self.pdf_inpath,
                        pages=str(curr_pgid),
                        flavor='stream',
                        strip_text='\n',
                        row_tol=15,
                        table_areas=[','.join([str(i)
                                              for i in flip_ori_curr_bbox])],
                        flag_size=True
                    )
                    if tables.n > 0:
                        table_df = tables[0].df
                        table_df = table_df.map(lambda x: re.sub(
                            CJK, '', x) if x else x)  # Remove Chinese characters
                        header = table_df.iloc[0].str.replace('\n', ' ')
                        if not any(i in [None, ''] for i in header.values.tolist()):
                            table_df.columns = header
                        else:
                            table_df.columns = self.last_table_header
                        self.last_table_header = table_df.columns
                        table_df = table_df[1:]
                        try:
                            table = json.loads(
                                table_df.to_json(orient="records"))
                            if table:
                                curr_txt = table
                        except ValueError:
                            pass
                except:
                    pass

            if curr_style == 'section':
                curr_style = section_rename(curr_txt)

            if curr_style != 'table':
                self.text += curr_txt

            # Rules to group or split elements into items
            if not prev:  # if first
                # create first item
                self.txt_blocks['content'].append({curr_style: curr_txt})
                ele_bboxes.update({curr_style: (ori_curr_bbox, curr_pgid)})
            else:
                assert len(
                    self.txt_blocks['content']) > 0, f"Text block content is empty. Cannot get the last text block item."
                last_txt_block = self.txt_blocks['content'][-1]
                curr_prev_sameStyle = prev_style == curr_style
                curr_prev_samePage = curr_pgid == prev_pgid

                # when current style is not the same as previous style
                if not curr_prev_sameStyle or prev_style == curr_style in ['section', 'sub_section', 'sub_sub_section', 'title', 'paragraph']:

                    # include the current text into current block if:
                    # 1. the current text block ISN'T far from previous text block by 50 units horizontally and vertically AND
                    # 2. current text has the same page id as previous one AND
                    # 3. shouldCurrInclude = True and shouldNextInclude = True base on inclusive element dictionary
                    # 4. previous style IS not a 'footer' or 'reference' (i.e. split block at 'footer' or 'reference') OR
                    # 5. current & previous style IS a paragraph and current text block has the same page id as previous one BUT different page id from next

                    shouldCurrInclude = False

                    inclusive_curr_tags = inclusive_with_prev[prev_style]
                    if curr_style in inclusive_curr_tags:
                        shouldCurrInclude = True

                    currIsNotSection = False
                    if self.document_type != 'esgReport':
                        if curr_style not in ['section', 'sub_section', 'sub_sub_section']:
                            currIsNotSection = True
                    # if current & previous text has same page id and current text should include in the same block
                    if curr_prev_samePage and shouldCurrInclude or currIsNotSection:
                        # continue updating the last item
                        # get number of style that match current style
                        dup_style_num = int(sum(curr_style == s.split(
                            '_')[0] for s in last_txt_block.keys()))
                        if dup_style_num > 0:
                            curr_style += '_' + str(dup_style_num)
                        ele_bboxes.update(
                            {curr_style: (ori_curr_bbox, curr_pgid)})
                        # split section: paragraph if string with such pattern found in previous section string but non-section current string
                        if self.document_type == 'termSheet' and prev_style == 'section' and re.match('(.*:)(.*)', last_txt_block['section']):
                            split = [i for i in re.match(
                                '(.*:)(.*)', last_txt_block['section']).groups() if i.strip()]
                            if len(split) == 2:
                                section, paragraph = split
                                if 'table' not in curr_style:
                                    if isinstance(curr_txt, list):
                                        if isinstance(curr_txt, list) and curr_style != 'table':
                                            curr_txt[0] = paragraph.strip(
                                            ) + ' ' + curr_txt[0]
                                        else:
                                            curr_txt.append(paragraph.strip())
                                    elif isinstance(curr_txt, dict):
                                        updict = {section: paragraph}
                                        updict.update(curr_txt)
                                        curr_txt = updict
                                    else:
                                        curr_txt = paragraph.strip() + ' ' + curr_txt
                            else:
                                section = split[-1]
                            last_txt_block.update({'section': section.strip()})
                        last_txt_block.update({curr_style: curr_txt})
                    else:
                        # otherwise append page no. to last item and create new item
                        # append current text to new item
                        item_bbox = bboxes_cluster_outline(tmp_bbox)
                        item_bbox = denormalize_bbox(
                            item_bbox, self.page_id2size[1]['width'], self.page_id2size[1]['height'])
                        # split section: paragraph if string with such pattern found in previous section string and section in current string
                        if self.document_type == 'termSheet' and curr_style == prev_style == 'section' and re.match('(.*:)(.*)', last_txt_block['section']):
                            split = [i for i in re.match(
                                '(.*:)(.*)', last_txt_block['section']).groups() if i.strip()]
                            if len(split) == 2:
                                section, paragraph = split
                                last_paragraph_style = 'paragraph'
                                dup_para_style_num = int(sum(last_paragraph_style == s.split('_')[
                                                         0] for s in last_txt_block.keys()))
                                if dup_para_style_num > 0:
                                    last_paragraph_style += '_' + \
                                        str(dup_para_style_num)
                                last_txt_block.update(
                                    {'section': section.strip()})
                                # create a new paragraph string in last text block
                                last_txt_block.update(
                                    {last_paragraph_style: paragraph.strip()})
                                ele_bboxes.update(
                                    {last_paragraph_style: (ori_curr_bbox, curr_pgid)})
                            else:
                                section = split[-1]

                        if isPageBreakOnSameStyleText:
                            item_bboxes.append(
                                {'value': item_bbox, 'page_id': prev_pgid})
                        last_txt_block.update(
                            {'page_id': page_id_range if isPageBreakOnSameStyleText else prev_pgid,
                             'id': item_id,
                             'bboxes_pageid': ele_bboxes,
                             'block_bbox': item_bboxes if isPageBreakOnSameStyleText else item_bbox})
                        ele_bboxes = {}
                        self.txt_blocks['content'].append(
                            {curr_style: curr_txt})
                        ele_bboxes.update(
                            {curr_style: (ori_curr_bbox, curr_pgid)})
                        item_id += 1
                        tmp_bbox = []
                        page_id_range = []
                        item_bboxes = []
                        isPageBreakOnSameStyleText = False
                # when current style is the same as previous style
                else:
                    style_duplicates_count = int(
                        sum(curr_style == s.split('_')[0] for s in last_txt_block.keys()))
                    if style_duplicates_count == 1:
                        prev_same_style = curr_style
                    elif style_duplicates_count > 1:
                        prev_same_style = curr_style + '_' + \
                            str(style_duplicates_count-1)
                    last_sameStyle_text = last_txt_block[prev_same_style]
                    if 'table' in curr_style and isinstance(last_sameStyle_text, list) and isinstance(curr_txt, list):
                        concat_text = last_sameStyle_text + curr_txt
                    elif 'table' in curr_style and isinstance(last_sameStyle_text, str) and isinstance(curr_txt, list):
                        concat_text = last_sameStyle_text + \
                            replace_char(curr['text'])
                    elif 'table' in curr_style and isinstance(last_sameStyle_text, list) and isinstance(curr_txt, str):
                        concat_text = prev_txt + curr_txt
                    else:
                        concat_text = last_sameStyle_text + " " + curr_txt
                    # update the last item with concatenating strings if current style equal to previous style and same page id
                    if curr_prev_samePage:
                        ele_bboxes.update(
                            {prev_same_style: (ori_curr_bbox, curr_pgid)})
                        last_txt_block.update({prev_same_style: concat_text})
                    # update the last item with concatenating strings if current style equal to previous style as 'paragraph' or 'list' but different page id
                    elif not curr_prev_samePage and curr_prev_sameStyle and curr_style in ['list', 'paragraph']:
                        isPageBreakOnSameStyleText = True
                        ele_bboxes.update(
                            {prev_same_style: ([ori_prev_bbox, ori_curr_bbox], [prev_pgid, curr_pgid])})
                        last_txt_block.update({prev_same_style: concat_text})
                        item_bbox = bboxes_cluster_outline(tmp_bbox)
                        tmp_bbox = []
                        item_bboxes.append(
                            {'value': item_bbox, 'page_id': prev_pgid})
                        if prev_pgid not in page_id_range:
                            page_id_range.append(prev_pgid)
                        if curr_pgid not in page_id_range:
                            page_id_range.append(curr_pgid)
                    # else update the last item bboxes page info, append current text to new item if current page no. not equal previous page id
                    else:
                        item_bbox = bboxes_cluster_outline(tmp_bbox)
                        if isPageBreakOnSameStyleText:
                            item_bboxes.append(
                                {'value': item_bbox, 'page_id': prev_pgid})
                        last_txt_block.update(
                            {'page_id': page_id_range if isPageBreakOnSameStyleText else prev_pgid,
                             'id': item_id,
                             'bboxes_pageid': ele_bboxes,
                             'block_bbox': item_bboxes if isPageBreakOnSameStyleText else item_bbox})
                        # reset temporary dictionaries and lists
                        ele_bboxes = {}
                        self.txt_blocks['content'].append(
                            {curr_style: curr_txt})
                        ele_bboxes.update(
                            {curr_style: (ori_curr_bbox, curr_pgid)})
                        item_id += 1
                        tmp_bbox = []
                        page_id_range = []
                        item_bboxes = []
                        isPageBreakOnSameStyleText = False
            if not nxt:  # if last
                last_txt_block = self.txt_blocks['content'][-1]
                if not tmp_bbox:
                    tmp_bbox.append(curr_bbox)
                item_bbox = bboxes_cluster_outline(tmp_bbox)
                last_txt_block.update(
                    {'page_id': prev_pgid,
                     'id': item_id,
                     'bboxes_pageid': ele_bboxes,
                     'block_bbox': item_bbox})
            tmp_bbox.append(curr_bbox)

    @timeit
    def transform_list_para(self):
        '''
        Transform list and paragraph element into numbering-content dictionary (for list) or phrase-split list (for paragraph)
        '''

        target_transform_elem = ['list', 'paragraph']
        list_para_content = [(item['id'], k, v) for item in self.txt_blocks['content']
                             for k, v in item.items() if any(ele in k for ele in target_transform_elem)]
        print('Multiprocessing list transformation with following setting: ')
        lst = multiprocess(batch_transform_dict_list, list_para_content, args=(
            self.document_type, self.nlp, self.override_key, self.txt_blocks))
        for block_id, txt_elem, text in lst:
            self.txt_blocks['content'][block_id].update({txt_elem: text})
        # for block_id, txt_elem, text in list_para_content:
        #     transform_dict_list((block_id, txt_elem, text), (self.document_type, self.nlp, self.override_key, self.txt_blocks))

        process_datetime = datetime.datetime.today().replace(tzinfo=pytz.utc)
        self.txt_blocks['process_datetime'] = process_datetime.strftime(
            "%d/%m/%Y, %H:%M:%S")

    @timeit
    def save_json(self):
        '''Save parsed segmentated and structures text into JSON document'''
        json_name = '.json'
        if self.txt_blocks:
            if not os.path.exists(self.json_outdir) and self.dev_mode:
                os.makedirs(self.json_outdir)
            if self.dev_mode:
                with open("{}{}{}".format(self.json_outdir, self.fname, json_name), 'w') as output:
                    json.dump(self.txt_blocks, output,
                              indent=4, ensure_ascii=False)
                print(
                    f'Document parsing output as JSON into path {self.json_outdir}{self.fname}{json_name}')
        else:
            raise ValueError(
                "The text block segmentation has not been done. Please run DocParser.txt_block_grp_seg()")

    @timeit
    def save_annot_order(self):
        ''' Annotate the order of text line parsed on pdf perform by pyMuPDF'''
        if self.annot_order:
            self.annot_order.save('{}{}_annot_order.pdf'.format(
                self.annot_outdir, self.fname))
        else:
            raise ValueError(
                "The document has not been annotated with parsing order. Please run DocParser.layoutlm_token_classify()")

    @timeit
    def save_annot_ele(self):
        '''Annotate the grouped token bboxes by element and id on pdf'''
        if self.use_model:
            txt_eles = [self.txt_ele, self.txt_model_ele]
        else:
            txt_eles = [self.txt_ele]
        pdf_names = ['_annot_ele.pdf', '_annot_model_ele.pdf']
        annot_eles = [self.annot_ele, self.annot_model_ele]

        for i in range(len(txt_eles)):
            if txt_eles[i]:
                for e in txt_eles[i]:
                    style = e['tag']
                    bbox = e['bbox']
                    page_id = e['page_id']
                    annot_eles[i] = annot_pdf_page(
                        annot_eles[i], page_id - 1, style, bbox, color=self.annot_style_color[style])
            else:
                raise ValueError(
                    "The text elements have not been classified. Please run DocParser.layoutlm_token_classify()")
            pdf_name = pdf_names[i]
            # Save annotated pdf
            annot_eles[i].save('{}{}{}'.format(
                self.annot_outdir, self.fname, pdf_name))

    @timeit
    def save_annot_txtblock(self):
        """ Annotate the pdf with bounding box to visual the group of items """
        pattern = '|'.join(
            [f'{i}' + r'(_){0,1}\d{0,}$' for i in self.ner_tags])
        if self.txt_blocks:
            for group in self.txt_blocks['content']:
                item_keys = list(group.keys())
                grouped_tags = [i for i in item_keys if re.match(pattern, i)]
                str_tags = " + ".join(grouped_tags)
                if isinstance(group['page_id'], int):
                    curr_page_id = group['page_id'] - 1
                    w = self.page_id2size[int(group['page_id'])]["width"]
                    h = self.page_id2size[int(group['page_id'])]["height"]
                    norm_bbox = normalize_bbox(group['block_bbox'], w, h)
                    self.annot_txtblock = annot_pdf_page(
                        self.annot_txtblock, curr_page_id,
                        "id: " + str(group['id']) + " " + str_tags + ' ' + str(norm_bbox), norm_bbox)
                elif isinstance(group['page_id'], list):
                    for i in group['block_bbox']:
                        curr_page_id = i['page_id'] - 1
                        w = self.page_id2size[int(i['page_id'])]["width"]
                        h = self.page_id2size[int(i['page_id'])]["height"]
                        item_bbox = i['value']
                        norm_bbox = normalize_bbox(item_bbox, w, h)
                        self.annot_txtblock = annot_pdf_page(
                            self.annot_txtblock, curr_page_id,
                            "id: " + str(group['id']) + " " + str_tags + ' ' + str(norm_bbox), norm_bbox)
        else:
            raise ValueError(
                "The text block segmentation has not been done. Please run DocParser.txt_block_grp_seg()")
        pdf_name = '_annot_txtblk.pdf'
        # Save annotated pdf
        self.annot_txtblock.save('{}{}{}'.format(
            self.annot_outdir, self.fname, pdf_name))

    def process_font_tag(self):
        '''Process document text font analysis and tagging'''
        self.fonts_line_spacing()
        self.word_spacing()
        self.font_tags()

    def process(self, do_segment=True):
        '''Process classification and text block segmentation, then save output and annotate pdf if do_annot=True'''
        if self.use_model:
            # if use model for prediction, create an img output folder for pdf2image
            create_folder(self.img_outdir)
            create_save_pdf_img(self.pdf_inpath, self.img_outdir, self.fname)

        self.extract_token()
        if self.use_model:
            self.layoutlm_token_classify()
            all_img_paths = [os.path.join(
                self.img_outdir, self.fname + f'_{str(i)}_ori.jpg') for i in range(1, self.page_num + 1)]
            for img_path in all_img_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
        self.token_level_grouping()
        if do_segment:
            print('Processing text block grouping ...')
            self.txt_block_grp_seg()
            print(
                'Completed processing text block grouping, transform list text to key-values ...')
            self.transform_list_para()
            print('Completed list transformation and phrase tokenization.')
        if self.dev_mode:
            self.save_output()
            if self.do_annot:
                self.annot_pdf(do_segment)

    def save_output(self):
        '''Save structured parsed text into text, JSON and csv files'''
        create_folder([self.json_outdir])
        self.save_json()

    # @timeit
    def annot_pdf(self, do_segment=True):
        '''Annotate text parsing order, element classification and text block segmentation results on pdf'''
        create_folder(self.annot_outdir)
        self.save_annot_order()
        # self.save_annot_ele()
        if do_segment:
            self.save_annot_txtblock()


def process_n_move_parsed(input, args):
    from pathlib import Path
    import timeit
    if isinstance(input, str):
        inpaths = [input]
    else:
        inpaths = input
    for inpath in inpaths:
        fname = os.path.basename(inpath)
        print("\nReport Name: ", fname)
        if args.document_type not in ['agreement', 'termSheet']:
            raise ValueError(
                "Document type input must be either 'esgReport' , 'agreement' or 'termSheet'.")
        parser = DocParser(inpath, args.output_json_dir, args.output_img_dir, args.use_model, args.use_ocr,
                           args.model_path, args.output_annot_pdf_dir, do_annot=args.do_annot, document_type=args.document_type)
        parser.process(do_segment=True)


def main():
    os.chdir(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pdf_dir",
        default=UNANNOTATED_PDF_DIR,
        type=str,
        required=False,
        help="The input pdf directory or single pdf. If it is a directory, it should contain the pdf files.",
    )
    parser.add_argument(
        "--output_json_dir",
        default=DOCPARSE_OUTPUT_JSON_DIR,
        type=str,
        required=False,
        help="The output directory where the output json data will be written.",
    )
    parser.add_argument(
        "--output_img_dir",
        default=OUTPUT_IMG_DIR,
        type=str,
        required=False,
        help="The output directory where to keep the pdf images.",
    )
    parser.add_argument(
        "--output_annot_pdf_dir",
        default=OUTPUT_ANNOT_PDF_DIR,
        type=str,
        required=False,
        help="The output directory where the output annotated pdf will be written.",
    )
    parser.add_argument(
        "--do_annot",
        default=False,
        type=bool,
        required=False,
        help="Set True to generate annotation on pdf for results visualization.",
    )
    parser.add_argument(
        "--use_model",
        default=USE_MODEL,
        type=bool,
        required=False,
        help="Set True to apply model for token classification prediction.",
    )
    parser.add_argument(
        "--model_path",
        default=DOCPARSE_MODELV3_CLAUSE_PATH,
        type=str,
        required=False,
        help="path to directory with tokens classification model.",
    )
    parser.add_argument(
        "--use_ocr",
        default=USE_OCR,
        type=bool,
        required=False,
        help="Set True to apply OCR on pdf to recognize text from image text.",
    )
    parser.add_argument(
        "--document_type",
        default=DOCPARSE_DOCUMENT_TYPE,
        type=str,
        required=False,
        help="Document type that pdf belongs to. Either 'agreement' or 'termSheet'",
    )
    doc_type = "FA"
    parser.add_argument(
        "--target_pdf_list",
        default=None,
        type=list,
        required=False,
        help="Target list of PDF filename in pdf_dir if specify else parse all pdf files in pdf_dir.",
    )

    args = parser.parse_args()
    if args.document_type == 'agreement':
        sub_folder_name = 'FA/'
    elif args.document_type == 'termSheet':
        sub_folder_name = 'TS/'
    args.output_json_dir = os.path.join(args.output_json_dir, sub_folder_name)

    # checks if path is a file
    isFile = os.path.isfile(args.pdf_dir)
    # checks if path is a directory
    isDirectory = os.path.isdir(args.pdf_dir)

    if isDirectory and args.document_type == 'termSheet':
        args.pdf_dir = os.path.join(args.pdf_dir, 'TS')
    elif isDirectory and args.document_type == 'agreement':
        args.pdf_dir = os.path.join(args.pdf_dir, 'FA')

    if isDirectory:
        # keep all pdf documents that going to be parsed in folder 'data/pdf'
        pdf_files = list(os.listdir(args.pdf_dir))
        pdf_files.sort()
        if args.target_pdf_list and isinstance(args.target_pdf_list, list):
            pdf_files = [os.path.join(
                args.pdf_dir, f) for f in pdf_files if f in args.target_pdf_list and f.endswith('.pdf')]
        else:
            # make sure extract files in .pdf only
            pdf_files = [os.path.join(args.pdf_dir, t)
                         for t in pdf_files if t.endswith('.pdf')]
        print("All Report Name: ", json.dumps([os.path.basename(
            i) for i in pdf_files], indent=4), f"\n no.of reports: {len(pdf_files)}")

        # multiprocess(process_n_move_parsed, pdf_files, args)
        for inpath in pdf_files:
            process_n_move_parsed(inpath, args)
        print("All Report Parsing Completed.")

    elif isFile:
        inpath = args.pdf_dir
        process_n_move_parsed(inpath, args)


if __name__ == '__main__':
    main()
