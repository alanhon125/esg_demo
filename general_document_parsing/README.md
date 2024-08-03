# Document Parsing

## Document Parsing with LayoutLM-large-uncased-500K-epoch_1 and rules

A rule and model hybrid tool for document parsing.

Built-in function of pyMUPDF to parse text and extract font characteristics (font size (float), font style flags (int), font name (str), color in sRGB format (int)).
which classify the font style into 3 elements types: heading, paragraph and subscript

append tag in front of text, with the following rules:

- paragraphs as text with

  1. the most frequently used font in the document,
  2. or same font size as the most frequently used font but with regular/light style and NOT same color
  3. or same font size as the most frequently used font but with NOT bold style and same color

- headers as any text
  1. LARGER than the most frequently used font
  2. or same font size as the most frequently used font but with NOT regular/light style and NOT same color
  3. or same font size as the most frequently used font but with bold style and same color
- subscripts as any text SMALLER the most frequently used font

It can generate a fine-grained font style with size id indicator after the tag, where the size of tag in order of following:
heading1< heading2 < heading3 <... < paragraph < subscript1 < subscript2 <...

With LayoutLM model token classification and rules, it further classifys the font style into 9 elements:

```
  1. 'caption':     texts surrounded to the pictures or tables
  2. 'figure':      pictures, figures or texts on charts in the documents
  3. 'footer':      page headers and footers
  4. 'list':        paragraph texts representing in bulletpoints
  5. 'paragraph':   texts in between section or cards
  6. 'reference':   reference texts in subscript style
  7. 'section':     headings and sub-heading describing the theme of paragraphs
  8. 'table':       texts in structured tables
  9. 'title':       theme or topic text
```

Based on the font characteristics of each parsed text in each block of each page,

1. Order the blocks of text based on layout (bounding box coordinate) with given single-column/multi-columns boolean indicator
2. Append an element tags <heading{size_id}>/< paragraph >/<subscript{size_id}> at the beginning of text and
3. Group the adjacent texts if they have the same element tag

Then return the list of parsed text

Given this list of tagged and ordered parsed text, group the text into items and save as JSON document with following schema:

```shell
{

    "filename":                         name_of_pdf,
    "page_num":                         num_of_pages_in_entire_doc,
    "items_num":                        num_of_items_in_entire_doc,
    "content": [
        {
            "element_tag_name":         "content_of_element",
            "element_position":         [start_char_in_entire_doc,end_char_in_entire_doc]
            "element_bbox":             normalized [x0,y0,x1,y1],
            ...,
            "page_id":                  page_id_of_item,
            "id":                       item_id,
            "item_bbox":                normalized [x0,y0,x1,y1],
            "parent_content":           [{
                                          "element1_tag_name":        "content_of_element1",
                                          "element2_tag_name":        "content_of_element2",
                                          ...,
                                          "page_id":                  page_id_of_item,
                                          "id":                       item_id,
                                        },...] or null,
            "children_num":             count_num_of_children,
            "child_id_range":           [start_child_id, end_child_id] or null,
            "child_page_range":         [start_child_page_id, end_child_page_id] or null
            "child_content":            [{
                                          "element1_tag_name":        "content_of_element1",
                                          "element2_tag_name":        "content_of_element2",
                                          ...,
                                          "page_id":                  page_id_of_item,
                                          "id":                       item_id,
                                        },...]
        },...]

}
```

### Installation

`pip install pymupdf`

### Source code

Download the document_parser.py from http://10.6.55.124/bigdata/esg_demo/edit/dev/esg-analytics/rule_based_processor/document_parser.py

Or

Download and run ESG document parser 2.ipynb

### Sample running

`document_parser.py`: program to parse general report in pdf and output structured text document and annotate text elements on documents (optional)

```shell
python document_parser.py --pdf_dir /path/to/pdf/directory \
                          --output_json_dir /path/to/data/output_json/directory \
                          --output_txt_dir /path/to/data/output_txt/directory \
                          --output_img_dir /path/to/data/output_img/directory \
                          --output_annot_pdf_dir /path/to/data/output_annotate_pdf/directory \
                          --do_annot \
                          --use_model \
                          --model_path /path/to/data/model/directory
```

where

- `pdf_dir` is the directory to store the input pdf documents.
- `output_json_dir` is the directory to store the output JSON documents.
- `output_txt_dir` is the directory to store the output .txt documents of parsed texts.
- `do_annot` set to True for making annotation (including annotation of token classifications, text parsing order, text block segmentation) on pdf; in case of `do_annot`, `output_annot_pdf_dir` have to be specified for location of storing annotated pdf
- `use_model` set to True for using model prediction on token classification with fine-tuned LayoutLM-large-uncased; in case of `use_model`, `model_path` and `output_img_dir` have to be specified for location of model and location of storing pdf page images that used by model respectively. (page images generated automatically when `use_model`)

############################################################

# Information extraction based on patterns and keywords

############################################################

# How to use: call info_ext_class.py and provide the text in the code

# The text can be a paragraph or multiple paragraphs

```python
python info_ext_class.py
```

    # Output: It will return a "text_info_ext" class object and have .text_data, .matchlist, and .result values:
    * 'text_data': the given text, in string format;
    * 'matchlist': all matched keywords in given 'text', in DataFrame format;
            keyword start_pos   end_pos
        0   1   61  61
        1   0.81    132 135
        2   kg  137 138
        3   CO2 143 145
        4   1   274 274
        ... ... ... ...
        17  non-hazardous waste 782 800
        18  13.41   852 856
        19  tonnes  858 863
        20  2019    1017    1020
        21  non-hazardous waste 1060    1078
    * 'result': all extracted information, in DataFrame format;
            number  num_start_pos   num_end_pos measurement meas_start_pos  meas_end_pos    keyword key_start_pos   key_end_pos
        0   0.81    132 135 kg  137 138 CO2 143 145
        1   0.325   327 331 kg  333 334 CO2 339 341
        2   0.061   710 714 kg  716 717 non-hazardous waste 722 740

## Training model

See `README.md` under `esg-analytics/general_document_parsing/layoutlmft` for details.

## PDF Pre-processing tool

A tool to convert pdf pages into images (output .jpg) for model training and inference, then extract texts from image (output text in .txt and annotated page in .jpg)

### Installation

`pip install pdf2image`
`pip install pdfplumber`
`pip install pdfminer`
`pip install Pillow`

### Source code

Download the document_parser.py from http://10.6.55.124/bigdata/esg_demo/edit/dev/esg-analytics/rule_based_processor/pdf_process.py

### Run pre-processing

```shell
python pdf_process.py   --data_dir /path/to/pdf/directory \
                        --output_dir /path/to/data/output/directory
```
