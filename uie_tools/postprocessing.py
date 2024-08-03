from api.config import DERIVED_DATA_CONFIG, METRIC_SCHEMA_CSV
import pandas as pd
import ast
import re

def convert_str_to_float(value):
    '''TODO: 20%
    '''
    # regex = r'-?\d+\.?\d*'
    regex = r'[-\d\.]+?'
    power_re = '(.+)<s>(.+)</s>'
    pattern = r'^(\d\.\d+)(\s?x?\s?10|e)(-?\d+)$' # 1.00 x 10-4, 1.00x10-4, 1.453e-4
    
    if not isinstance(value, str):
        return float(value)
    else:
        find = re.search(pattern, value)
        if find:
            string = re.sub(pattern, r'\1E\3', value)
            number = float(string)
            return number
        else:
            try:
                if re.search('\.', value):
                    number = float(value)
                else:
                    number = int(value)
                return number
            except ValueError:
                if re.search('<s>', value):
                    try:
                        m, n = re.findall(power_re, value)[0]
                        if re.search('x', m):
                            m1, m2 = re.split('x', m)
                            m1 = m1.strip()
                            m2 = m2.strip()
                            number = float(m1) * pow(int(m2), int(n))
                        else:
                            if re.search('\.', m):
                                number = float(''.join(re.findall(regex, m)))
                            else:
                                number = int(''.join(re.findall(regex, m)))
                        return number
                    except Exception as e1:
                        if re.search(regex, value):
                            try:
                                if re.search('\.', value):
                                    number = float(''.join(re.findall(regex, value)))
                                else:
                                    number = int(''.join(re.findall(regex, value)))
                                return number
                            except Exception as e1:
                                return ''
                                print(f'{value} cannot be converted to be numeric, error: {e1}')
                        else:
                            return ''
                            print(f'{value} cannot be converted to be numeric, error: {e1}')
                else:
                    if re.search(regex, value):
                        try:
                            if re.search('\.', value):
                                number = float(''.join(re.findall(regex, value)))
                            else:
                                number = int(''.join(re.findall(regex, value)))
                            return number
                        except Exception as e3:
                            return ''
                            print(f'{value} cannot be converted to be numeric, error: {e3}')
                    else:
                        return ''
                        print(f'{value} cannot be converted to be numeric')


def convert_unit_and_value(unit, value):
    unit_re = "10<s>(\d+)</s>" # 10<s>4</s> m<s>3</s>
    converted_value = convert_str_to_float(value)
    if re.search(unit_re, unit):
        subscript = int(re.findall(unit_re, unit)[0])
        multiplier = pow(10, subscript)
        split_unit = re.split(unit_re, unit)[-1]
        converted_unit = re.sub('<s>|</s>|\n', '', split_unit)
        if converted_value:
            converted_value = converted_value * multiplier
    else:
        converted_unit = re.sub('<s>|</s>|\n', '', unit)
    # TODO: M e t r i c   t o n s   p e r
    # converted_unit = u' '.join([u''.join(re.split(' ', item)) for item in re.split(' ', converted_unit)])
    # converted_value = converted_value.strip()
    converted_unit = converted_unit.strip()
    if not converted_value:
        converted_value = 0
    return converted_unit, converted_value


def calc_derived_data(derivation_info, table_metrics_records):
    ''' derivation_info: 
        {
            "metric": "GHG Scope 1 Intensity (per employee)",
            "source": "either",
            "fomular": "GHG Scope 1/ Number of employee"
        }
    ''' 
    env_unit_df = pd.read_csv(METRIC_SCHEMA_CSV)
    metric_unit_map = {}
    for item in env_unit_df.to_dict(orient='records'):
        for metric in ast.literal_eval(item['metric']):
            try:
                if isinstance(item['unit'], str):
                    if re.search('\[', item['unit']):
                        metric_unit_map[metric.lower()] = ast.literal_eval(item['unit'])[0]
                    else:
                        metric_unit_map[metric.lower()] = item['unit']
                else:
                    pass
            except Exception as e:
                print(e, item['unit'])
    
    derived_metric = derivation_info['metric'].lower()
    
    # years = list(set([item['year'] for item in table_metrics_records if item['target_metric'].lower() == derived_metric]))
    years = list(set([item['year'] for item in table_metrics_records]))
    derived_items = []
    
    for year in years:
        derived_item = dict()
        derived_item['year'] = year
        if re.search('/', derivation_info['fomular']):
            dividend, divisor = re.split('/', derivation_info['fomular'])
            dividend_value = [
                item['value'] for item in table_metrics_records if (item['target_metric'].lower() == dividend.strip().lower())
                & (item['year'] == year)
            ]
            divisor_value = [
                item['value'] for item in table_metrics_records if (item['target_metric'].lower() == divisor.strip().lower())
                & (item['year'] == year)
            ]
            if dividend_value:
                if divisor_value:
                    derived_item['value'] = dividend_value[0] / divisor_value[0]
        elif re.search('\+', derivation_info['fomular']):
            sum_value = 0
            addend_list = re.split('\+', derivation_info['fomular'])
            num_of_addend = len(addend_list)
            count = 0
            for addend in addend_list:
                addend_value = [
                    item['value'] for item in table_metrics_records if (item['target_metric'].lower() == addend.strip().lower())
                    & (item['year'] == year)
                ]
                if addend_value:
                    count += 1
                    sum_value += addend_value[0]
            if count == num_of_addend:
                derived_item['value'] = sum_value
        if derived_item.get('value'):
            derived_item['source'] = "derived"
            derived_item['metric'] = derived_item['target_metric'] = derivation_info['metric']
            derived_item['unit'] = derived_item['converted_unit'] = metric_unit_map.get(derived_metric)
            derived_item['company_name'] = table_metrics_records[0]['company_name']
            derived_item['position'] = table_metrics_records[0]['position']
            derived_item['converted_value'] = derived_item['value']
            derived_item['type'] = 'derived'

            derived_items.append(derived_item)
    return derived_items


def gen_derived_data(table_metrics_records):
    metrics = list(set(item['metric'] for item in table_metrics_records))
    derivation = []
    for item in DERIVED_DATA_CONFIG:
        if item['source'] == 'either':
            if item['metric'] in metrics:
                pass
            else:
                derived_items = calc_derived_data(item, table_metrics_records)
                print(derived_items)
                derivation.extend(derived_items)
        elif item['source'] == 'derived':
            derived_items = calc_derived_data(item, table_metrics_records)
            derivation.extend(derived_items)
    return derivation
