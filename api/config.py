from postgresql_storage.db_util import DbUtil

# Input Directory
PDF_DIR = 'data/pdf/'

# Output Directories
DOCPARSE_OUTPUT_JSON_DIR = 'data/docparse_json/'
METRIC_OUTPUT_JSON_DIR = 'data/text_metric_json/'
REASONING_OUTPUT_JSON_DIR = 'data/text_reasoning_json/'
TABLE_OUTPUT_JSON_DIR = 'data/table_metric_json/'
OUTPUT_TXT_DIR = 'data/txt/'
OUTPUT_ANNOT_PDF_DIR = 'data/annot_pdf/'
OUTPUT_TABLE_DETECTION_DIR = 'data/tables/'
OUTPUT_IMG_DIR = 'data/image/'
OUTPUT_PRED_IMG_DIR = 'data/pred_img/'
OUTPUT_CSV_DIR = 'data/csv/'
OUTPUT_LAYOUTLM_INPUT_DIR = 'data/layoutlm_input_data/'
OUTPUT_LAYOUTLM_OUTPUT_DIR = 'data/layoutlm_output_data/'
LOG_DIR = 'data/log/'

# Database Tables
METRIC_EXTRACTION_TABLE_NAME = "metric_entity_relation_unfiltered"
REASONING_EXTRACTION_TABLE_NAME = "reasoning_entity_relation_unfiltered"
ENV_TABLE_NAME = 'test_environment_info'
METRICS_TABLENAME = "test_environment_info"
METRICS_TABLENAME2 = "test_environment_info"
PDFFILES_TABLENAME = 'pdffiles'
PDFFILES_INFO_TABLENAME = 'pdffiles_info'
METRIC_SCHEMA_TABLENAME = "metric_schema"

# Target Schema Tables
METRIC_SCHEMA_CSV = 'data/schema/target_metric.csv'
TARGET_ASPECT_TABLE = 'data/schema/target_aspect.csv'

# Models and Model setting
USE_MODEL = True
DOCPARSE_MODEL_PATH = 'models/checkpoints/layoutlm/layoutlm_base_500k_docbank_epoch_1'
DOCPARSE_MODELV3_PATH = 'models/checkpoints/layoutlm/layoutlmv3_large_500k_docbank_epoch_1_lr_1e-5_1407_esg_epoch_2000_lr_1e-5'
DOCPARSE_MODELV3_CLAUSE_PATH = 'models/checkpoints/layoutlm/layoutlmv3_base_500k_docbank_epoch_2_lr_1e-5_511_agreement_epoch_2000_lr_1e-5'
DOCPARSE_MODELV3_TS_PATH = 'models/checkpoints/layoutlm/layoutlmv3_base_500k_docbank_epoch_2_lr_1e-5_511_agreement_epoch_2000_lr_1e-5_347_termsheet_epoch_2000_lr_1e-5'
TABLE_DETECT_MODEL_PATH = 'table_transformer/pubtables1m_detection_detr_r18.pth'
DOCPARSE_BATCH_SIZE = 16

# UIE Models and Model setting
# Please refer to models/config.yaml for UIE model configurations
UIE_ROOT_PATH = '/home/data1/public/ResearchHub/UIE/'
UIE_METRIC_MODELV1_PATH = UIE_ROOT_PATH + 'output_store/astri/astri_esg_108_split/meta_2022-07-25-15-51-18162_hf_models_uie-base-en_spotasoc_astri_astri_esg_108_split_e50_linear_lr5e-4_ls0_b32_wu0.06_n-1_RP_sn0.1_an0.1_run5'
UIE_RESONING_MODELV1_PATH = UIE_ROOT_PATH + 'output_store/astri/esg_reasoning_split-0803/meta_2022-08-03-10-53-20697_hf_models_uie-base-en_spotasoc_astri_esg_reasoning_split_e50_linear_lr5e-4_ls0_b32_wu0.06_n-1_RP_sn0.1_an0.1_run5'
UIE_BATCH_SIZE = 128 # max no. of sentences per batch input

# GPU allocation for model inference
TABLE_EXTRACT_GPU_ID = "3"
DOCPARSE_GPU_ID = "0,1,2,3"

# Document Parsing output configuration
DOCPARSE_PDF_ANNOTATE = False
DOCPARSE_DOCUMENT_TYPE = 'agreement'

# Target elements for UIE
TARGET_ELEMENTS = ['paragraph', 'list', 'caption']

# Lists and Dictionary for key metric and KPI keywords
KEY_ENV_WORDS = [
    "ENVIRONMENTAL", "ENVIRONMENTAL ASPECTS", "ENVIRONMENTAL MANAGEMENT",
    "ENVIRONMENTAL PROTECTION",
    "EMISSIONS", "Emission Management", "performance"
]
KEY_PERFORMANCE_KPI_SCHEMA = {
    'air_emissions': ['Nitrogen Oxide', 'Sulphur Oxide', 'Particulate Matter'],
    'GHG_emissions': ['GHG Scope 1 Emissions', 'GHG Scope 2 Emissions', 'Total GHG Emissions'],
    'hazardous_waste': ['Hazardous Wastes'],
    'non-hazardous_waste': ['Non-hazardous Wastes']
}
NOT_KEY_ENV_REGEX = r'^Environmental, Social and Governance Report$'

# External API URLs
EXTERNAL_API_HOST_IP = "10.6.55.126"  # "10.6.55.3" for aiserver, "10.6.55.126" for data-dl-1
URL_METRIC_V1 = f'http://{EXTERNAL_API_HOST_IP}:8002/api/v1/metrics_predictor/predict'
URL_METRIC_V3 = f'http://{EXTERNAL_API_HOST_IP}:8002/api/v3/metrics_predictor/predict'
URL_REASONING = f'http://{EXTERNAL_API_HOST_IP}:8002/api/v1/reasoning_predictor/predict'

# derived data
DERIVED_DATA_CONFIG = [
    {
        "metric": "GHG Scope 1 Emissions Per Employee",  # derived metric name
        "source": "either",  # either / derived
        "fomular": "GHG Scope 1 Emissions / GHG Scope 1 Emissions",  # fomular for derived metrics calculation
    }
]

TEST_INFO = {
    "PERSTA_Environmental,SocialandGovernanceReport2020.pdf": [21],
    "新海能源_Environmental,SocialandGovernanceReportforYear2020.pdf": [38],
    "中石化煉化工程_2020Environmental,SocialandGovernanceReport.pdf": [26],
    "山東墨龍_2020Environmental,SocialandGovernanceReport.pdf": [23],
    "飛尚無煙煤_2020Environmental,SocialandGovernanceReport.pdf": [4, 5],
    "中國石油股份_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf": [75, 76],
    "中石化油服_2020Environmental,Social,andGovernance(ESG)Report.pdf": [19, 35],
    "中能控股_Environmental,SocialandGovernanceReport2020.pdf": [33, 34],
    "匯力資源_2020Environmental,SocialandGovernanceReport.pdf": [7, 8, 10],
    "西伯利亞礦業_Environmental,SocialandGovernanceReport2020.pdf": [4, 5, 6],
    "金泰能源控股_Environmental,SocialandGovernanceReport2020.pdf": [6, 7, 8, 9],
    "兗煤澳大利亞_ESGReport2020.pdf": [22, 31, 32, 33],
    "惠生工程_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf": [32, 62, 63, 64],
    "易大宗_Environmental,SocialandGovernanceReport2020.pdf": [33, 34, 35, 36],
    "上海石油化工股份_2020CorporateSocialResponsibilityReport.pdf": [6, 7, 29, 34, 36, 37],
    "中信資源_2020ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf": [76, 77, 78, 79, 80, 81, 82],
    "中國海洋石油_2020Environmental,SocialandGovernanceReport.pdf": [31, 32, 33, 47],
    "中國石油化工股份_2020SinopecCorp.SustainabilityReport.pdf": [19, 29, 43, 44, 45],
    "中煤能源_ChinaCoalEnergyCSRReport2020.pdf": [19, 21, 25, 29, 31, 33, 37],
    "元亨燃氣_Environmental,socialandgovernancereport2020_21.pdf": [16, 17, 18, 26, 27, 28],
    "兗礦能源_SocialResponsibilityReport2020OfYanzhouCoalMiningCompanyLimited.pdf": [55, 70, 71, 72, 73],
    "南南資源_Environmental,SocialandGovernanceReport2020_21.pdf": [20, 21, 22, 23, 32, 33],
    "巨濤海洋石油服務_Environmental,SocialandGovernanceReport2020.pdf": [12, 13, 15, 16, 19, 20],
    "延長石油國際_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf": [10, 12, 15, 16, 18],
    "海隆控股_2020Environmental,SocialandGovernanceReport.pdf": [37, 38, 39, 40, 41, 42],
    "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2020.pdf": [10, 21, 24, 25, 26, 28, 30, 34],
    "陽光油砂_2020Environmental,SocialandGovernanceReport.pdf": [8, 9, 10, 11, 13, 14, 15, 18],
    "安東油田服務_2020SUSTAINABILITYREPORT.pdf": [7, 25, 39, 45, 51, 73, 76, 77, 78],
    "中國神華_2020Environmental,ResponsibilityandGovernanceReport.pdf": [75, 77, 78, 84, 95, 96, 97, 98],

    # TODO: need to check this
    "上海石油化工股份_2021CorporateSocialResponsibilityReport.pdf": [6, 7, 31, 38, 39],
    "中信資源_2021ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT.pdf": [42, 43, 44],
    "中國石油化工股份_2019SinopecCorp.CommunicationonProgressforSustainableDevelopment.pdf": [40, 41, 42],
    "中石化煉化工程_2021Environmental,SocialandGovernanceReport.pdf": [21, 23, 27, 43, 44],
    "易大宗_Environmental,SocialandGovernanceReport2021.pdf": [48, 49, 50, 51, 54],
    "蒙古能源_ENVIRONMENTAL,SOCIALANDGOVERNANCEREPORT2021.pdf": [13, 24, 26, 27, 30, 32],

    "00003.pdf": [51, 52],
    "00008.pdf": [55, 56],
    "00009.pdf": [30],
    # "00017.pdf": [57,58],
    # "00012.pdf": [68,69,70],
    "00019.pdf": [157],
    "00034.pdf": [30],
    "00030.pdf": [8, 10, 11],
    "00027.pdf": [47],
    "00029.pdf": [8, 10, 11],

    "boc_03988_2021.pdf": [57, 58, 59, 61],
    "tecent_00700_2021.pdf": [124, 132, 133, ],
    "中石化_00386_2021.pdf": [50, 51],
    "环球大通_00905_Environmental,SocialandGovernanceReport2021.pdf": [8, 9, 10, 11, 15],
    "領悅服務集團_02165_Environmental,Social,andGovernanceReport2021.pdf": [15, 16]

}

# Character blocks
DIACRITICS = r'[\u0300-\u036F]+'
FULLWIDTH_ASCII_VARIANTS = r'[\uff01-\uff5e]+'
CJK = r'[\u4e00-\u9fff]+'

# Regular Expression for sentence or phrase splits on paragraph in document parsing
COMMA_SPLIT_PATTERN = r'((?<!\d{3}),(?! \bwhich\b)(?! \band\b)(?! \bincluding\b))'
PARA_SPLIT_PATTERN = r'( and/or;|; and/or| and;|; and| or;|; or|;)'
SECTION_SPLIT_AT_COMMA = [
    "Documentation", "Amendments and Waivers", "Miscellaneous Provisions", "Other Terms"]

# Regex for extraction and split numbering from text into key:value pairs

XYZ = r'\b[xyz]\b'
CAPITAL_XYZ = r'\b[XYZ]\b'
CAPITAL_ROMAN_NUM = r"\b(?=[XVI])M*(X[L]|L?X{0,2})(I[XV]|V?I{0,3})\b"
ROMAN_NUM = r"\b(?=[xvi])m*(x[l]|l?x{0,2})(i[xv]|v?i{0,3})\b"
ONE_CAPITAL_ALPHABET = r'\b[^\d\sIVXivxa-zXYZ\W_]{1}\b'
ONE_LOWER_ALPHABET = r'\b[^\d\sIVXivxA-Zxyz\W_]{1}\b'
TWO_ALPHABET = r'\b[^\d\sIVXivxA-Zxyz\W_]{2}\b'
ONE_DIGIT = r'\b[1-9]\d{0,1}\b'
TWO_DOTS_DIGIT = rf'{ONE_DIGIT}\.{ONE_DIGIT}\.{ONE_DIGIT}'
ONE_DOT_DIGIT = rf'{ONE_DIGIT}\.{ONE_DIGIT}'
DOT_DIGIT = rf'{TWO_DOTS_DIGIT}|{ONE_DOT_DIGIT}'
DELIMITERS = [
    'to :',
    ';'
]
SYMBOLS = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐',
           '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖', '_']
NUMBERING_LIST = [XYZ, CAPITAL_XYZ, CAPITAL_ROMAN_NUM, ROMAN_NUM,
                  ONE_CAPITAL_ALPHABET, ONE_LOWER_ALPHABET, TWO_DOTS_DIGIT, ONE_DOT_DIGIT, ONE_DIGIT]
ALPHABET_LIST = ['[a-z]', '[a-z]{2}', '[a-z]{3}', '[a-z]{4}',
                 '[a-z]{5}', '[A-Z]', '[A-Z]{2}', '[A-Z]{3}', '[A-Z]{4}', '[A-Z]{5}']
numWithBrack1 = ['\(' + i + '\)' for i in ALPHABET_LIST]
numWithBrack2 = ['\(' + i + '\) and' for i in ALPHABET_LIST]
numWithBrack3 = ['\(' + i + '\),' for i in ALPHABET_LIST]
numWithBrack4 = ['\(' + i + '\) to' for i in ALPHABET_LIST]
numWithBrack5 = ['\(' + i + '\) or' for i in ALPHABET_LIST]
numWithBrack = numWithBrack2 + numWithBrack3 + \
    numWithBrack4 + numWithBrack5  # numWithBrack1 +
p = inflect.engine()
numInWord = [p.number_to_words(i).capitalize() for i in range(100)] + [p.number_to_words(i) for i in range(100)] + [
    '\['+p.number_to_words(i).capitalize()+'\]' for i in range(100)] + ['\['+p.number_to_words(i)+'\]' for i in range(100)]
PREPOSITIONS = [
    '\babout\b',
    '\babove\.\b',
    '\babove\)\b',
    '\babove\b',
    '\bacross\b',
    '\band\b',
    '\bat\b',
    '\bbelow\.\b',
    '\bbelow\)\b',
    '\bbelow\b',
    '\bbesides\b',
    '\bby\b',
    '\bfor\b',
    '\bfrom\b',
    '\bin\b',
    '\bincrease\b',
    '\bof\b',
    '\bon\b',
    '\bor\b',
    '\breduce\b',
    '\bthan\b',
    '\bto\b',
    '\bunder\b',
    '\bwith\b',
    '\bwithin\b',
    # ' to ',
    '\bafter\b',
    'and\\\or'
]

PUNCT_ALL_STR = re.escape(string.punctuation)
PURE_PUNCT_DIGIT = rf'^[{PUNCT_ALL_STR}]+$|^[\d]+$'
PUNCT_LIST = [re.escape(i)
         for i in string.punctuation if i not in ['(', '[', '"', '$', ':']]

NLB_COMMON = [
    "<",
    ">",
    "\&",
    "\(\d{1}\) \bto\b",
    "\(\d{2}\) \bto\b",
    "\(\w{1}\) \bto\b",
    "\(\w{2}\) \bto\b",
    "added",
    "Agreement",
    "agreement",
    "aircraft",
    "Aircraft",
    "Article",
    "article",
    "Articles",
    "articles",
    "Basel",
    "Basel",
    "BORROWER",
    # "Borrower",
    "clause",
    "Clause",
    "clause\(s\)",
    "clauses",
    "Clauses",
    "Column",
    "column",
    "Columns",
    "columns",
    "Company",
    "company",
    "Counsel",
    "counsel",
    "CRD",
    # "equal to",
    "Facility",
    "facility",
    "General",
    "general",
    "greater than",
    "Guarantor",
    "guarantor",
    "in the case of",
    "In the case of",
    "KPI",
    "less than",
    "limbs",
    "para",
    "Paragraph",
    "paragraph",
    "Paragraph\(s\)",
    "paragraph\(s\)",
    "Paragraphs",
    "paragraphs",
    "Plan",
    "plan",
    "Premium",
    "premium",
    "Property",
    "property",
    "Proviso",
    "proviso",
    "Section",
    "section",
    "Sections",
    "sections",
    "Shares",
    "shares",
    "Ship",
    "ship",
    "sub-paragraph",
    "Sub-paragraph",
    "sub-paragraphs",
    "Sub-paragraphs",
    "Tranche",
    "tranche",
    "Unit",
    "unit",
    "Vessel",
    "vessel"
]

# negative lookbehind list (list of patterns that should not be preceded by numbering) for numbering with bracket,
# e.g not to extract numbering with pattern Clause (a), Paragraph (e) etc.
NLB_NUM_BRACK = [
    "\(\w[\)|\.]\s\band\b",
    "\w[\)|\.]\s\band\b",
    "within"
] + NLB_COMMON

NLB_BRACK = NLB_NUM_BRACK + numInWord + numWithBrack

# negative lookbehind list (list of patterns that should not be preceded by numbering) for numbering with dot,
# e.g not to extract numbering with pattern Clause 1., Paragraph e. etc.
NLB_NUM_DOT = [
    #   "\[",
    "\d",
    '[A-Z]\.',
    "at least",
    "at most",
    "exceed"
] + NLB_COMMON

NLB_DOT = NLB_NUM_DOT + numInWord + numWithBrack

# negative lookahead list (list of patterns that is not present immediately after numbering) for numbering with dot, e.g not to extract and split numbering with pattern 1.0, 2.1 years etc.
# Asserts that what immediately follows the current position in the string is not in this list
NLA_NUM_DOT = [
    ' \babove\b',
    '\bSc\b',
    '\bSc\b\.',
    '[a-zA-Z]\.',
    # '\d{1}',
    '\d{1} \%',
    '\d{1}\%',
    '\d{1} \(',
    '\d{1} per cent',
    '\d{1} percent',
    '\d{1} Years',
    '\d{1} years',
    '\d{1} yrs',
    '\d{1}\([a-zA-Z]\)',
    '\d{1}x',
    # '\d{2}',
    '\d{2} \%',
    '\d{2}\%',
    '\d{2} \(',
    '\d{2} per cent',
    '\d{2} percent',
    '\d{2} Years',
    '\d{2} years',
    '\d{2} yrs',
    # '\d{2}\([a-zA-Z]\)',
    'x'] + PUNCT_LIST
# negative lookahead list (list of patterns that is not present immediately after numbering) for numbering with bracket, e.g not to extract and split numbering with pattern 1)., (2)& etc.
NLA_NUM_BRACK = PUNCT_LIST + PREPOSITIONS
