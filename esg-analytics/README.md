# ESG_Demo

Demo system for environmental metric extraction from ESG reports

## Getting started

When working with Python, it’s highly recommended to use a virtual environment. This prevents dependency collisions and undesired behaviors that would otherwise arise from premature or unintended version changes. As such, we highly encourage you to set up a virtual environments using conda for the Anaconda Python distribution.

### Check conda is installed and in your PATH

Open a terminal client.

Enter `conda -V` into the terminal command line and press enter.

If conda is installed you should see somehting like the following.

```bash
$ conda -V
conda 3.7.0
```

If not, please install Anaconda Distribution from [here](https://www.anaconda.com/products/distribution).

### Installing requirements

```bash
conda create -n esg_demo python=3.7
conda activate esg_demo
git clone http://10.6.55.124/bigdata/esg_demo.git
cd ~/esg_demo
pip install -r requirements.txt
```

### Installing poppler¶

Poppler is the underlying project that does the magic in pdf2image that uses to convert pdf into images.
You can check if you already have it installed by calling `pdftoppm -h` in your `terminal/cmd`.

#### Ubuntu

`sudo apt-get install poppler-utils`

#### Archlinux

`sudo pacman -S poppler`

#### MacOS

`brew install poppler`

#### Windows

Download the latest package from [here](http://blog.alivate.com.au/poppler-windows/)

Extract the package

Move the extracted directory to the desired place on your system

Add the bin/ directory to your PATH

Test that all went well by opening cmd and making sure that you can call `pdftoppm -h`

# Deploy Django REST API

## Create the PostgreSQL Database

We will be setting up a PostgreSQL database. We have already provided and create a database models that use for our ESG Django application in `esg_demo/esg-analytics/postgresql_storage/`

## Initialization a Django Project

Activating models that we've defined in `esg_demo/esg-analytics/postgresql_storage/models.py`

```commandline
python manage.py makemigrations
```

You should see something similar to the following:

```commandline
Migrations for 'postgresql_storage':
  postgresql_storage/migrations/0001_initial.py
    - Create model EnvironmentInfo
    - Create model MetricSchema
    - Create model MetricEntityRelation
    - Create model PdfFiles
    - Create model ReasoningEntityRelation
    - Create model TestEnvironmentInfo
    - Create model TestTextInfo
```

Now, we can migrate the initial database schema to our PostgreSQL database using the management script:

```commandline
python manage.py migrate
```

You should see something similar to the following:

```commandline
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, postgresql_storage, sessions
Running migrations:
  Applying postgresql_storage.0001_initial... OK
```

## Start the development server

Run the following command to interact with the API in localhost in your local machine:

```commandline
python manage.py runserver 0.0.0.0:8000
```

## Start the Django server for ESG demo system

with log (2>&1: redirect stderr to stdout) and no hang up. Port 8000 for testing & debugging; Port 8001 for demo system

```bash
conda activate esg_demo
cd /home/data1/public/ResearchHub/esg_demo/esg-analytics
nohup python manage.py runserver 0.0.0.0:8000 > data/log/syslog_port8000.log 2>&1 &
nohup python manage.py runserver 0.0.0.0:8001 > data/log/syslog_port8001.log 2>&1 &
```

## Start the Django server for UIE

with log (2>&1: redirect stderr to stdout) and no hang up. Port 8002 for universal information extraction API

```bash
conda activate esg_demo
cd /home/data1/public/ResearchHub/esg_demo/esg-analytics/models
nohup python manage.py runserver 0.0.0.0:8002 > ../data/log/syslog_port8002.log 2>&1 &
```

## Testing API with postman application

Also, you'll want to download the free version of Postman. Postman is an excellent tool for developing and testing APIs.
If you already have a postman, go ahead and test the API following the procedures below. But if you don’t have it installed on your local machine, then click
[here](https://www.postman.com/downloads/) to download it.

# Usage

Here we provide APIs for:

1. General Document Parsing
2. General Table Extraction
3. General Text Entity Relation Extraction

Please see the following usages examples:

## General Document Parsing

Given the filename of ESG report (that pdf have already been upload to esg_demo/esg-analytics/data/pdf), response the structured textual document of report with elements, layout positions, parent & childs labelled.

Folder `esg_demo/esg-analytics/general_document_paring` includes codes for the document parsing module.

Server's view page URL on server: http://0.0.0.0:8000/api/generate_doc_parser_result

Prerequisite: You need to upload ESG report in pdf to folder `esg_demo/esg-analytics/data/pdf`,
the filename should in format `"<COMPANY_NAME>_<DESCRIPTION_WITH_YEAR_INCLUDED>.pdf"`, in which the year of report must be expressed in "yyyy" format.

Get the response by GET method:

the input value should be a string of filename that going to be parsed(Here the input filename MUST consistent with the uploaded document name):

Input:

```json
{
  "filenames": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf"
}
```

and you will get the response as following:

```json
{
    "success": true,
    "parsed_doc": {
        "filename": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport",
        "page_num": 12,
        "items_num": 47,
        "content": [
            {
                "title": "2020 ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT Feishang Anthracite Resources Limited \u98db\u5c1a\u7121\u7159\u7164\u8cc7\u6e90\u6709\u9650\u516c\u53f8",
                "title_position": [
                    0,
                    96
                ],
                "title_bbox": [
                    138,
                    845,
                    623,
                    898
                ],
                "footer": "(Incorporated in the British Virgin Islands with limited liability) Stock Code : 1738",
                "footer_position": [
                    97,
                    181
                ],
                "footer_bbox": [
                    138,
                    903,
                    535,
                    934
                ],
                "page_id": 1,
                "id": 0,
                "item_bbox": [
                    138,
                    607,
                    623,
                    934
                ],
                "parent_info": null,
                "children_num": 0,
                "child_id_range": null,
                "child_page_range": null,
                "child_content": null
            }, ...
        ]
    }
```

Meanwhile, the parsed document will be output as JSON document in `esg_demo/esg-analytics/data/docparse_json`

## General Table Extraction

Folder `esg_demo/esg-analytics/general_table_extraction` includes codes for the table information extraction module.

Server's view page URL on server: http://0.0.0.0:8000/api/generate_key_metrics

Prerequisite: You need to upload ESG report in pdf to folder `esg_demo/esg-analytics/data/pdf`,
the filename should in format `"<COMPANY_NAME>_<DESCRIPTION_WITH_YEAR_INCLUDED>.pdf"`, in which the year of report must be expressed in "yyyy" format.

Get the response by GET method:

the input value should be a string of filename that going to be parsed(Here the input filename MUST consistent with the uploaded document name):

Input:

```json
{
  "filename": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf",
  "mode": "ESG"
}
```

and you will get the response as following:

```json
{
    "success": true,
    "result": [
        {
            "company_name": "飛尚無煙煤",
            "raw_metrics": "Nitrogen Oxide (NOx)",
            "metrics": "Nitrogen Oxide (NOx)",
            "similar_score": 1.000000238418579,
            "unit": "kg",
            "year": "2019",
            "value": "1.57"
        },
        {
            "company_name": "飛尚無煙煤",
            "raw_metrics": "Nitrogen Oxide (NOx)",
            "metrics": "Nitrogen Oxide (NOx)",
            "similar_score": 1.000000238418579,
            "unit": "kg",
            "year": "2020",
            "value": "1.66"
        },
        {
            "company_name": "飛尚無煙煤",
            "raw_metrics": "Sulphur Oxide (SOx)",
            "metrics": "Sulphur Oxide (SOx)",
            "similar_score": 1.0000001192092896,
            "unit": "kg",
            "year": "2019",
            "value": "59.36"
        },
        {
            "company_name": "飛尚無煙煤",
            "raw_metrics": "Sulphur Oxide (SOx)",
            "metrics": "Sulphur Oxide (SOx)",
            "similar_score": 1.0000001192092896,
            "unit": "kg",
            "year": "2020",
            "value": "56.80"
        },
        {
            "company_name": "飛尚無煙煤",
            "raw_metrics": "Particulate Matter (PM)",
            "metrics": "Particulate Matter (PM)",
            "similar_score": 1.0,
            "unit": "kg",
            "year": "2019",
            "value": "4.37"
        },
        {
            "company_name": "飛尚無煙煤",
            "raw_metrics": "Particulate Matter (PM)",
            "metrics": "Particulate Matter (PM)",
            "similar_score": 1.0,
            "unit": "kg",
            "year": "2020",
            "value": "4.18"
        }, ...
      ]
}
```

## General Text Entity Relation Extraction

Folder `esg_demo/esg-analytics/models` includes codes for the UIE text information extraction module. It also includes the **micro server config** to run this API standalone.

We run the sentence-level inference of text metric entity relation extraction at AI server(10.6.55.3) in order to boost Deep Learning Inference with view page URL: http://10.6.55.3:8002/api/v1/metrics_predictor/predict
We also run the sentence-level inference of text reasoning entity relation extraction at AI server(10.6.55.3) in order to boost Deep Learning Inference with view page URL: http://10.6.55.3:8002/api/v1/reasoning_predictor/predict

The doc-level inference of text metric entity relation extraction running at server's view page URL on server: http://0.0.0.0:8000/api/generate_text_entity_relation

Prerequisite: You need to upload ESG report in pdf to folder `esg_demo/esg-analytics/data/pdf`,
the filename should in format `"<COMPANY_NAME>_<DESCRIPTION_WITH_YEAR_INCLUDED>.pdf"`, in which the year of report must be expressed in "yyyy" format.

Get the response by GET method:

the input value could be either a string of filename (Here the input filename MUST consistent with the uploaded document name):

Input:

```json
{
  "filenames": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf"
}
```

or a list of filenames

```json
{
  "filenames": [
    "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf",
    "飛尚無煙煤_2021Environmental,SocialandGovernanceReport.pdf"
  ]
}
```

and you will get the response if input as a string of a filename:

```json
{
    "success": true,
    "results": {
        "filename": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport",
        "company_name": "飛尚無煙煤",
        "year": "2020",
        "metric_entity_relations": [
            {
                "page_id": 4,
                "text_block_id": 11,
                "paragraph": [
                    {
                        "sent_id": 6,
                        "sentence": "For the year ended 31 December 2020, air emissions, including nitrogen oxides, sulphur oxides and particulate matter, were mainly produced from the Company\u2019s vehicles many of which weighing 2",
                        "ners": [
                            {
                                "metric": "air emissions",
                                "metric_char_position": [
                                    37,
                                    49
                                ],
                                "target_metric": "Total Carbon Emissions",
                                "similarity": 0.5299999713897705,
                                "relation": "equal",
                                "number": "2",
                                "number_char_position": [
                                    190,
                                    190
                                ],
                                "value": "2",
                                "unit": null
                            }, ...
                        ],
                       "split_sentence": [
                           {
                               "text": "For the year ended 31 December 2020, ",
                               "type": "normal",
                               "char_position": [
                                   0,
                                   37
                               ]
                           },
                           {
                               "text": "air emissions",
                               "type": "metric",
                               "char_position": [
                                   37,
                                   49
                               ]
                           },...
                        ]
                    }, ...
                ]
            },...
        ],
        "reasoning_entity_relations": [
            {
                "page_id": 2,
                "text_block_id": 2,
                "paragraph": [
                    {
                        "sent_id": 7,
                        "sentence": "Based on the comments from the corporate social responsibility committee, the Board has evaluated the ESG risks to be low as the Group has complied with all relevant laws and regulations in all material aspects.",
                        "ners": [
                            {
                                "head_entity_type": "target",
                                "head_entity": "ESG risks",
                                "head_entity_char_position": [
                                    102,
                                    110
                                ],
                                "target_aspect": null,
                                "similarity": 0.0,
                                "relation": "comply_with",
                                "tail_entity_type": "guideline",
                                "tail_entity": "all relevant laws and regulations",
                                "tail_entity_char_position": [
                                    153,
                                    185
                                ],
                            }
                        ],
                        "split_sentence": [
                           {
                               "text": "Based on the comments from the corporate social responsibility committee, the Board has evaluated the ",
                               "type": "normal",
                               "char_position": [
                                   0,
                                   102
                               ]
                           },
                           {
                               "text": "ESG risks",
                               "type": "target",
                               "char_position": [
                                   102,
                                   111
                               ]
                           },...
                        ]
                    }
                ]
            },...
        ]
}
```

or you will get the response if input as a list of filenames:

```json
{
  "success": true
}
```

### Output Description

1. The document parsing output kept as JSON document in `esg_demo/esg-analytics/data/docparse_json`,
   Please see `README.md` under `esg_demo/esg-analytics/general_document_parsing` for details of the output.

2. The metric entity-relation extraction kept as JSON document in `esg_demo/esg-analytics/data/text_metric_json`.
   The output with JSON schema:

```json
{
    "filename": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport",
    "company_name": "飛尚無煙煤",
    "year": "2020",
    "metric_entity_relations": [
        {
            "page_id": 4,
            "text_block_id": 11,
            "paragraph": [
                {
                    "sent_id": 6,
                    "sentence": "For the year ended 31 December 2020, air emissions, including nitrogen oxides, sulphur oxides and particulate matter, were mainly produced from the Company’s vehicles many of which weighing 2",
                    "ners": [
                        {
                            "metric": "air emissions",
                                "metric_char_position": [
                                  [
                                    37,
                                    49
                                  ]
                                ],
                                "target_metric": "Total Carbon Emissions",
                                "similarity": 0.5299999713897705,
                                "relation": "equal",
                                "number": "2",
                                "number_char_position": [
                                  [
                                    190,
                                    190
                                  ]
                                ],
                                "value": "2",
                                "unit": null
                        }
                    ]
                }, ...
            ], ...
        }, ...
    ]
```

3. The reasoning entity-relation extraction kept as JSON document in `esg_demo/esg-analytics/data/text_reasoning_json`.
   The output with JSON schema:

```json
{
    "filename": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport",
    "company_name": "飛尚無煙煤",
    "year": "2020",
    "reasoning_entity_relations": [
            {
                "page_id": 2,
                "text_block_id": 2,
                "paragraph": [
                    {
                        "sent_id": 7,
                        "sentence": "Based on the comments from the corporate social responsibility committee, the Board has evaluated the ESG risks to be low as the Group has complied with all relevant laws and regulations in all material aspects.",
                        "ners": [
                            {
                                "head_entity_type": "target",
                                "head_entity": "ESG risks",
                                "head_entity_char_position": [
                                  [
                                    102,
                                    110
                                  ]
                                ],
                                "relation": "comply_with",
                                "tail_entity_type": "guideline",
                                "tail_entity": "all relevant laws and regulations",
                                "tail_entity_char_position": [
                                  [
                                    153,
                                    185
                                  ]
                                ],
                            }
                        ]
                    }
                ]
            },...
        ]
```

### Configurations

See the Wiki for more details.

### Features

Support English text strings,

- [ ] currently NOT support Chinese.
