# General Text Extraction with parsed pdf document

UIE's Text Extraction combined with document parsing Deployment

## Usage

This is an API to perform (or query) document parsing and metric entity-relation extraction in a stream.

### Install Requirements

```bash
git clone http://10.6.55.124/bigdata/esg_demo.git
cd esg-analytics
pip install -r requirements.txt
```

### Launch the Server

You may deploy the server directly at server 10.6.55.243 with

```bash
cd ~/esg_demo/esg-analytics
python3 manage.py runserver 10.6.55.243:8000
```

Server's view page URL on server: http://10.6.55.243:8000/api/generate_metric_entity_relation

### Server response

Prerequisite: You need to upload ESG report in pdf to folder `10.6.55.243/home/data/esg_demo/esg-analytics/data/pdf`, 
              the filename should in format `"<COMPANY_NAME>_<DESCRIPTION_WITH_YEAR_INCLUDED>.pdf"`, in which the year of report must be expressed in "yyyy" format.

Get the response by GET method:

the input value could be either a string of filename (Here the input filename MUST consistent with the uploaded document name):

```json
{
  "filenames": "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf"
}
```

or a list of filenames

```json
{
  "filenames": ["飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf",
                "飛尚無煙煤_2021Environmental,SocialandGovernanceReport.pdf"]
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
                                      [
                                        102,
                                        110
                                      ]
                                    ],
                                    "target_aspect": null,
                                    "similarity": 0.0,
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

1. The document parsing output kept as JSON document in `10.6.55.243/home/data/esg_demo/esg-analytics/data/docparse_json`, 
Please see `README.md` under `esg-analytics/rule_based_processor` for details of the output.

2. The metric entity-relation extraction kept as JSON document in `10.6.55.243/home/data/esg_demo/esg-analytics/data/text_metric_json`. 
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
            ], ...
        }, ...
```
3. The reasoning entity-relation extraction kept as JSON document in `10.6.55.243/home/data/esg_demo/esg-analytics/data/text_reasoning_json`. 
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
                                "target_aspect": null,
                                "similarity": 0.0,
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
```