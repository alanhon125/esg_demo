def word_pos(sent, string):
    '''
    search the string in a given sentence and return the nearest starting character position and ending character position
    if the string is not found in the sentence, return None

    @param sent: a string of sentence for reference
    @type sent: str
    @param string: a string that to be examine its character position in a sentence
    @type string: str
    @rtype: list
    @return: list of [starting character position, ending character position] or None
    '''
    import re
    pos = []
    escaped = re.escape(string)
    p = re.compile(escaped)
    for m in p.finditer(sent):
        pos.append([m.start(), m.end()])
    return pos


def has_numbers(inputString):
    '''
    detect if the input string contains any digit character

    @param inputString: a string that to be examinate its existence of digit character
    @type inputString: str
    @rtype: boolean
    @return: True if contains any digit character
    '''
    return any(char.isdigit() for char in inputString)


def split_sentence(sentence, entities):
    '''
    split sentence on given list of entities, return list of sentence fragment in order excluding those strings of entities
    @param sentence: a string that to be split
    @type sentence: list
    @param entities: list of [a string that sentence split at, [start_char, end_char]]
    @type entities: list
    @rtype: list
    @return: list of sentence fragments exclude entities
    '''
    splits = sentence
    while entities:
        # print(f'entities: {entities}')
        search = entities[0]
        search_count = 0
        for frag in sentence:
            # print(f'frag: {frag}, search_count: {search_count}')
            if frag.find(search) != -1:
                # print(f'match {search} in {frag}')
                index = sentence.index(frag)
                splits.remove(frag)
                entities.remove(entities[0])
                start, end = word_pos(frag, search)[0]
                splits.insert(index, frag[:start])
                splits.insert(index + 1, frag[end:])
                sentence = splits
                break
            search_count += 1
            if search_count >= len(sentence):
                # print(f'{search} not found in {sentence}')
                entities.remove(entities[0])
                break
    return splits


def scientific_notation_to_float(string):
    import re
    pattern = r'^(\d\.\d+)(\s?x|X|e\s?10?)(-?\d+)$'  # 1.00 x 10-4, 1.00x10-4, 1.453e-4
    find = re.search(pattern, string.lower())
    if find:
        string = re.sub(pattern, r'\1E\3', string)
        string = float(string)
    else:
        try:
            string = float(string)
        except ValueError:
            print(f"Cannot convert string {string} into float")
    return string


def get_relation_numeric_unit(no_with_unit):
    '''
    split no_with_unit into value and unit, also identify the comparative relation (more than, less than, approximately equal, equal)
    Example: get_relation_numeric_unit('more than 50 m3/day') >> ('more than', '50', 'm3/day')
    @param no_with_unit: a string that to be examinate its existence of digit character
    @type no_with_unit: str
    @rtype: tuple
    @return: relation, value value, unit
    '''
    import re
    converter_big_num = {'ten': 10,
                         'hundred': 100,
                         'thousand': 1000,
                         'ten thousand': 10000,
                         'hundred thousand': 100000,
                         'million': 1000000,
                         'ten million': 10000000,
                         'hundred million': 100000000,
                         'billion': 1000000000,
                         'ten billion': 10000000000,
                         'hundred billion': 10000000000,
                         'trillion': 1000000000000,
                         'ten trillion': 10000000000000,
                         'hundred trillion': 10000000000000
                         }
    no_with_unit = re.sub(r'(\b\d{2})(,)(\d{2}\b)', r'\1.\3', no_with_unit)  # 21,95 -> 21.95
    no_with_unit = re.sub(',', '', no_with_unit)  # replace all , to empty string
    for k, v in converter_big_num.items():
        big_num_matcher = re.search(r'(\D*)(\d*\.{0,1}\d+ {0,1}-{0,1})(%s)(.*)' % k, no_with_unit,
                                    re.I)  # 1.50 million, $1.5 million tons/year
        if big_num_matcher:
            unit1 = big_num_matcher.group(1)  # $
            value1 = float(big_num_matcher.group(2).split('-')[0])  # str(1.50) -> float(1.50)
            value2 = converter_big_num[big_num_matcher.group(3).lower()]  # million -> 1000000
            unit2 = big_num_matcher.group(4)  # tons/year
            no_with_unit = unit1 + str(value1 * value2) + unit2  # $1.5 million tons/year -> $1500000 tons/year
        # no_with_unit = re.sub(r' {0,1}-{0,1}%s' % k, v, no_with_unit) # 150 million -> 150000000
    converter_small_num = {
        '%': 1 / 100,  # percent
        '‰': 1 / 1000,  # per mille
        '‱': 1 / 10000  # per ten thousand
    }
    for k1, v1 in converter_small_num.items():
        small_num_matcher = re.search(r'(\D*)(\d+\.{0,1}\d*)( {0,1}%s)(.*)' % k1,
                                      no_with_unit)  # 1.50%, $1.5% tons/year, RMB 1.50 %
        if small_num_matcher:  # if '%' is found
            unit1 = small_num_matcher.group(1)  # $
            value1 = float(small_num_matcher.group(2)) * v1  # str(1.50) -> float(1.50)*1/100 -> 0.015
            unit2 = small_num_matcher.group(4)  # tons/year
            no_with_unit = unit1 + str(value1) + unit2
    no_with_unit = re.sub('\(|\)|\{|\}|\[|\]', '', no_with_unit)  # replace all brackets to empty string
    no_with_unit = re.sub(r"(\d+)( to )(\d+)", r'\1-\3', no_with_unit)  # 21 to 95 -> 21-95
    specialChar = r'[^A-Za-z0-9$€£￥\-~]'
    currencySymbol = r"\w{0,3}\$|\€|\£|\￥"  # US$, RMB￥
    pattern = r'(\D*)([\d\,*\.]*\W*[\d\,*\.]+)(\D*\d*\D*\d*)'  # 10,000-20,000% / RMB10,000
    g = list(re.match(pattern, no_with_unit.strip()).groups())
    if re.match(specialChar, g[0].strip()):
        g[0] = ''

    prefix = str(g[0].strip().lower())
    suffix = str(g[2].strip().lower())

    prefix_suffix = [prefix, suffix]

    comparative = {
        '^no less than': 'more than',
        '^no more than': 'less than',
        '^over|more than|greater than|or above': 'more than',
        '^less than|smaller than|or below|up to|at most': 'less than',
        '^approximately|nearly|about|almost|around|~': 'approximately equal'
    }
    relation = 'equal'
    for k, v in comparative.items():
        flag = False
        for i, string in enumerate(prefix_suffix):
            if i == 1:
                i = 2
            if re.search(k, string):
                match = re.search(k, string)[0]
                relation = v
                if i == 0:
                    g[i] = ''
                else:
                    g[i] = re.sub(match, '', g[i])
                flag = True
                break
        if flag == True:
            break

    if '-' in g[1].strip() or '~' in g[1].strip():  # if value contains - or ~
        value = re.sub(r'(\d+)([^0-9]*)(\d*)([^0-9]*)', r'\1-\3',
                       g[1].strip())  # concatenate two numerical values with -
    else:
        value = g[1].strip()
    value = scientific_notation_to_float(value)

    for string in prefix_suffix:
        if re.search(currencySymbol, string):
            unit = re.search(currencySymbol, string)[0]
    if 'unit' not in locals():
        if g[0].strip() and g[2].strip():
            unit = g[2].strip()
        else:
            unit = (g[0].strip() or g[2].strip())
    unit = re.sub(r'^-{0,1}(t)( \.*)', r'tons\2', unit)  # if t find in unit, replace with tons
    # unit = re.sub(r'\btons\b', 'tonnes', unit)  # if tonnes find in unit, replace with tons
    # unit = re.sub(r'\bton\b', 'tonne', unit)
    return relation, value, unit


def get_similarity(test_string_list, ref_string_list, sim_threshold=0.6):
    '''
    calculate string similarity between test string and reference list of strings using Word2Vec,
    return (the most similar string from reference list, similarity score) which greater than given similarity threshold.

    @param test_string_list: a string that to be examinate its similarity to reference list of string
    @type test_string_list: str
    @param ref_string_list: a reference list of list of strings
    @type ref_string_list: list
    @param sim_threshold: a similarity score threshold to accept the max similarity score and return string (by default = 0.6)
    @type sim_threshold: float
    @rtype: tuple
    @return: tuple of (the most similar string from first item of reference list, similarity score)
    '''
    from gensim.models import Word2Vec
    ref_words_list = []
    for data_field_list in ref_string_list:
        for data_field in data_field_list:
            if data_field or data_field != '':
                ref_words = data_field.lower().split()
                if ref_words not in ref_words_list:
                    ref_words_list.append(ref_words)
    test_words_list = test_string_list.lower().split()
    model = Word2Vec(sentences=test_words_list + ref_words_list, vector_size=100,
                     window=5, min_count=1, workers=4)
    all_sims = []
    for data_field_list in ref_string_list:
        sims = []
        for data_field in data_field_list:
            if data_field or data_field != '':
                data_field = data_field.lower().split()
                similarity = model.wv.n_similarity(data_field, test_words_list)
                sims.append(similarity)
        max_sim = max(sims)
        all_sims.append(max_sim)

    max_all_sims = max(all_sims)
    max_index = all_sims.index(max_all_sims)

    relevant = False
    if max_all_sims >= sim_threshold:
        relevant = True
        sim_score = max_all_sims
        ref_string = ref_string_list[max_index][0]

    if not relevant:
        return None, 0
    else:
        return ref_string, round(sim_score, 2)


def get_similarity_sentbert(test_string_list, ref_string_list, sim_threshold=0.6,
                            model_path='models/checkpoints/all-MiniLM-L6-v2', pretrained=None,
                            return_single_response=True):
    '''
    calculate string similarity between test string and reference list of strings using SentenceTransformer,
    return (the most similar string from reference list, similarity score) which greater than given similarity threshold.

    @param test_string_list: a string or list of string that to be examinate its similarity to reference list of string
    @type test_string_list: list or str
    @param ref_string_list: a reference list of list of strings OR string
    @type ref_string_list: list of list or str
    @param sim_threshold: a similarity score threshold to accept the max similarity score and return string (by default = 0.6)
    @type sim_threshold: float
    @param model_path: the local model path of Sentence-BERT
    @type model_path: str
    @param pretrained: if prediction leverage pre-trained model from huggingface, provide the name of model, otherwise set to None
    @type pretrained: str
    @param return_single_response: set True if single response of (string, similarity score) as a result, otherwise response (list of string, list of similarity score)
    @type return_single_response: boolean
    @rtype: tuple
    @return: tuple of (the most similar string from first item of reference list, similarity score)
    '''
    from sentence_transformers import SentenceTransformer, util
    import torch

    if pretrained is not None:
        model = SentenceTransformer(pretrained, device=torch.device("cuda", 2))
    else:
        model = SentenceTransformer(model_path, device=torch.device("cuda", 2))

    if isinstance(test_string_list, str):
        test_words_list = test_string_list.lower()
        embeddings1 = model.encode(test_words_list, convert_to_tensor=True)

        all_sims = []
        relevant = False

        if isinstance(ref_string_list, list) or isinstance(ref_string_list, tuple):
            for data_field_list in ref_string_list:
                sims = []
                for data_field in data_field_list:
                    if data_field or data_field != '':
                        data_field = data_field.lower()
                        embeddings2 = model.encode(data_field, convert_to_tensor=True)
                        similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
                        sims.append(similarity)
                max_sim = max(sims)
                all_sims.append(max_sim)

            if return_single_response:
                max_all_sims = max(all_sims)
                max_index = all_sims.index(max_all_sims)

                if max_all_sims >= sim_threshold:
                    relevant = True
                    sim_score = round(max_all_sims, 2)
                    ref_string = ref_string_list[max_index][0]
            else:
                ref_string = []
                sim_score = []
                for sim in all_sims:
                    if sim >= sim_threshold:
                        above_threshold_idx = all_sims.index(sim)
                        relevant = True
                        sim_score.append(round(sim, 2))
                        ref_string.append(ref_string_list[above_threshold_idx][0])

        else:
            embeddings2 = model.encode(ref_string_list, convert_to_tensor=True)
            sim_score = round(util.pytorch_cos_sim(embeddings1, embeddings2).item(),2)
            if sim_score >= sim_threshold:
                relevant = True
                ref_string = ref_string_list

        if not relevant:
            return None, 0
        else:
            return ref_string, sim_score

    elif isinstance(test_string_list, list) or isinstance(test_string_list, tuple):
        # test_words_list = [s.lower() for s in test_string]
        test_words_list = test_string_list
        schema_list = []
        schema_map = {}
        similar_pairs = {}
        for data_field_list in ref_string_list:
            for data_field in data_field_list:
                if data_field or data_field != '':
                    data_field = data_field.lower()
                    schema_list.append(data_field)
                    schema_map.update({data_field: data_field_list[0]})
        embeddings1 = model.encode(test_words_list, convert_to_tensor=True)
        embeddings2 = model.encode(schema_list, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        for i in range(0, len(test_words_list)):
            if return_single_response:
                max_index = list(cosine_scores[i]).index(max(cosine_scores[i]))
                max_all_sims = max(cosine_scores[i]).item()
                if max_all_sims >= sim_threshold:
                    sim_score = round(max_all_sims, 2)
                    ref_string = schema_map.get(schema_list[max_index])
                    # else:
                    #     sim_score = 0
                    #     ref_string = None

                    similar_pairs.update({
                        test_words_list[i]: {
                            'similar_metrics': ref_string,
                            'score': sim_score
                        }
                    })
            else:
                all_score = list(cosine_scores[i])
                above_threshold_idx = [all_score.index(k) for k in [j for j in all_score if j >= sim_threshold]]
                above_threshold_sims = [j.item() for j in all_score if j >= sim_threshold]
                idx_sims = list(zip(above_threshold_idx, above_threshold_sims))
                idx_sims = sorted(idx_sims, key=lambda x: x[1], reverse=True)
                ref_string = []
                sim_score = []
                for idx, sims in idx_sims:
                    string = schema_map.get(schema_list[idx])
                    if string not in ref_string:
                        ref_string.append(string)
                        sim_score.append(round(sims, 2))

                if ref_string:
                    similar_pairs.update({
                        test_words_list[i]: {
                            'similar_metrics': ref_string,
                            'score': sim_score
                        }
                    })

        return similar_pairs


def is_grammarly_incorrect(tool, text):
    import gc
    # get the matches
    matches = tool.check(text)
    has_error = len(matches) > 0
    del matches
    gc.collect()
    return has_error


def split_value_unit(string, task='information_extraction', schema=['Unit'], model='uie-m-base'):
    '''
    extract unit of measurement of string using PaddleNLP Taskflow API and model uie-m-base (https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/taskflow.md)

    @param string: a string of number that contain value and unit
    @type string: str
    @rtype: tuple
    @return: tuple of value, unit that split from string of number
    '''
    from pprint import pprint
    from paddlenlp import Taskflow
    import re

    ie = Taskflow(task, schema=schema, model=model, schema_lang="en")
    res = ie([string])
    converter_big_num = {'ten': 10,
                         'hundred': 100,
                         'thousand': 1000,
                         'ten thousand': 10000,
                         'hundred thousand': 100000,
                         'million': 1000000,
                         'ten million': 10000000,
                         'hundred million': 100000000,
                         'billion': 1000000000,
                         'ten billion': 10000000000,
                         'hundred billion': 10000000000,
                         'trillion': 1000000000000,
                         'ten trillion': 10000000000000,
                         'hundred trillion': 10000000000000
                         }
    converter_small_num = {
        '%': 1 / 100,  # percent
        '‰': 1 / 1000,  # per mille
        '‱': 1 / 10000  # per ten thousand
    }
    try:
        unit = res[0]['Unit'][0]['text']
        try:
            if string.split(unit)[0].strip():  # if value exists in upper part of number string (e.g. 260,000 tonnes)
                value = string.split(unit)[0].strip()
            else:  # if value exists in lower part of number string (e.g. RMB 1.6 billion)
                value = string.split(unit)[1].strip()
            value = re.sub(',', '', value)
            for k, v in converter_big_num.items():
                if k in value:
                    value = re.sub(k, '', value)
                    value = float(value) * v
                    break
            value = float(value)
        except:
            print(f'Value {string.split(unit)[0].strip()} in the string {string} cannot be convert into float')
            return None, unit
    except:
        print(f'Unit is not found in string {string}')
        try:
            for k1, v1 in converter_small_num.items():
                if k1 in string:
                    value = float(re.sub(k1, '', string))
                    value = value * v1
                    print(f'Value {str(value)} is found in string {string}')
                    return value, None
                    break
        except:
            print(f'Neither Value not Unit could be found in string {string}')
        return None, None
        # pprint(res)
    return value, unit

def uie_reasoning_validator(head_entity_type, head_entity, tail_entity_type, tail_entity, relation):
    valid_schema = [
                {
                    'entity_type_1': 'target',
                    'entity_type_2': 'method',
                    'relation': ['increased_by','reduced_by']
                },
                {
                    'entity_type_1': 'target',
                    'entity_type_2': 'guideline',
                    'relation': ['comply_with']
                }
            ]
    if not head_entity and not tail_entity or head_entity in tail_entity or tail_entity in head_entity:
        return False
    for schema in valid_schema:
        if head_entity_type == schema['entity_type_1']:
            if tail_entity_type == schema['entity_type_2']:
                if relation in schema['relation']:
                    return True
        elif tail_entity_type == schema['entity_type_1']:
            if head_entity_type == schema['entity_type_2']:
                if relation in schema['relation']:
                    return True
    return False