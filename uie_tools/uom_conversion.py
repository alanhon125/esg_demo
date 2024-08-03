from uie_tools.utils import get_similarity_sentbert
from postgresql_storage.metric_extraction_models import MetricSchema
import ast
import re

def si_unit_convert(source_uom, target_uom, target_metric_density):
    '''
    calculate magnitude of multiplier to convert from source unit of measurement (uom) to target unit of measurement
    if both source uom and target uom are SI unit and return full name of target uom

    @param source_uom: a string of source unit of measurement
    @type source_uom: str
    @param target_uom: a string of target unit of measurement
    @type target_uom: str
    @param target_metric_density: density of target metric substance in kg/m3 if applicable
    @type target_metric_density: float
    @rtype: tuple
    @return: tuple of (multiplier for conversion, full name of target uom)
    '''
    import pint
    u = pint.UnitRegistry()
    Q = u.Quantity

    source = Q(1, source_uom)
    if target_metric_density:
        target1 = Q(1, target_uom)
        source_uom_type = str(source.dimensionality)
        target_uom_type = str(target1.dimensionality)
        if source_uom_type == '[mass]' and target_uom_type == '[length] ** 3': # from mass to volume
            target2 = source.to('kg')
            multiplier = target2.m / target_metric_density
            target = Q(multiplier, 'cubic meter').to(target_uom)
            print(f'Convert from {source_uom}【{source_uom_type}】 to {target_uom}【{target_uom_type}】by multiplying {multiplier}')
        elif source_uom_type == '[length] ** 3' and target_uom_type == '[mass]': # from volume to mass
            target2 = source.to('cubic meter')
            multiplier = target_metric_density * target2.m
            target = Q(multiplier, 'kg').to(target_uom)
            print(f'Convert from {source_uom}【{source_uom_type}】 to {target_uom}【{target_uom_type}】by multiplying {multiplier}')
        elif source_uom_type == target_uom_type:
            print(f'Same type of unit convert from {source_uom}【{source_uom_type}】 to {target_uom}【{target_uom_type}】')
            target = source.to(target_uom)
        else:
            return None, None
    else:
        target = source.to(target_uom)

    multiplier = target.m
    target_uom_name = target.u
    print("=================== SI unit conversion ===================")
    print(f'source_value =【{source.m:{"^"}{10}{"f"}}】 | source_uom =【{source.u}】')
    print(f'target_value =【{target.m:{"^"}{10}{"f"}}】 | target_uom =【{target.u}】\n')
    return multiplier, target_uom_name

def currency_convert(source_curr, target_curr, year):
    '''
    calculate magnitude of multiplier to convert from source unit of measurement (uom) to target unit of measurement
    if both source uom and target uom are currency unit

    @param source_curr: a string of source currency unit of measurement (must be in ISO 4217 Currency alpha codes format, e.g. USD, CNY etc.)
    @type source_curr: str
    @param target_curr: a string of target currency unit of measurement (must be in ISO 4217 Currency alpha codes format, e.g. USD, CNY etc.)
    @type target_curr: str
    @param year: a string of year that currency rate refer to
    @type year: str
    @rtype: float
    @return: multiplier for currency conversion
    '''
    from currency_converter import CurrencyConverter
    from datetime import date
    c = CurrencyConverter()
    multiplier = c.convert(1, source_curr, target_curr, date=date(int(year), 12, 31))
    print("=================== Currency unit conversion ===================")
    print(f'【{source_curr} : {target_curr}】 = 【1 : {multiplier}】 at year end of【{year}】\n')
    return multiplier

def uom_2_target_uom(uom, target_uom, metric_year, target_metric_density):
    # try to match uom with list of target uom, return uom if matched
    multiplier = 1
    uom_alike, sim_score = get_similarity_sentbert(uom, target_uom)
    if sim_score >= 0.8:
        print(f'Successfully match uom:【{uom}】based on SentenceBERT similarity matching with target uom:【{target_uom}】with similarity score =【{sim_score}】')
        return multiplier, uom_alike

    # try to convert uom with si unit
    try:
        multiplier, target_uom_name = si_unit_convert(uom, target_uom, target_metric_density)
        print(f'Successfully convert uom:【{uom}】based on SI target uom:【{target_uom}】')
        return multiplier, target_uom
    except:
        pass

    # try to convert currency uom
    try:
        multiplier = currency_convert(uom, target_uom, metric_year)
        print(f'Successfully convert currency uom:【{uom}】based on currency target uom:【{target_uom}】')
        return multiplier, target_uom
    except:
        return None, None

def convert_arbitrary_uom(uom, target_metric, metric_year, metric_schema=MetricSchema, uom_colname='unit'):
    '''
    convert arbitrary unit of measurement with given source string of uom and list of target uom

    @param uom: a string of source unit of measurement in any format
    @type uom: str
    @param target_metric: a string of target metric
    @type target_metric: str
    @param metric_year: a string of year that metric refer to
    @type metric_year: str
    @rtype: tuple
    @return: tuple of (multiplier for conversion, target uom)
    '''
    target_uom = metric_schema.objects.filter(
        metric__icontains=target_metric).values(uom_colname).first()[uom_colname]
    target_metric_density = metric_schema.objects.filter(
        metric__icontains=target_metric).values('density_NTP_kg_per_m3').first()['density_NTP_kg_per_m3']

    if not target_uom:
        target_uom = None
    else:
        if target_uom.startswith('["'):
            target_uom = ast.literal_eval(target_uom)
            target_uom = target_uom[0]
            target_uom = re.sub('/', ' per ', target_uom)

    uom = re.sub('/', ' per ', uom) # Substitute "/" into " per ", e.g. tCO2e/tonne -> tCO2e per tonne

    print(f'target_metric = 【{target_metric:{"^"}{40}{"s"}}】 | source_uom = 【{uom:{"^"}{10}{"s"}}】 | target_uom = 【{target_uom}】')
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
    uom_expression = {
        r'm {0,1}3' : 'cubic meters',
        r'm {0,1}2' : 'square meters',
        r'hk {0,1}\$' : 'HKD',
        r'us {0,1}\$' : 'USD',
        ('rmb','yuan','¥',r'CN {0,1}¥') : 'CNY',
        ('tCO2','tCO2e','tonne of carbon dioxide equivalent','tonne of CO2 equivalent') : 'tCO2-e',
        ' per h' : ' per hour'
    }
    currency_pattern = r'([\u0024\u20AC\u00A5A-Z\s]{0,4})([0-9.,]*)([\s\u0024\u20AC\u00A5A-Z]{0,4})' # € 55.78 | 5,854.78USD | 854.78 ¥

    # try to map uom into standard uom expression
    for k, v in uom_expression.items():
        if isinstance(k,str):
            uom = re.sub(k, v, uom, flags=re.I)
        else:
            for i in k:
                uom = re.sub(i, v, uom, flags=re.I)

    # try to split uom into base uom and uom divisor if target_uom contains ' per ' (i.e. intensity uom)
    if target_uom:
        if ' per ' in target_uom:
            target_uom_base , target_uom_divisor = target_uom.split(' per ')
            for k, v in converter_big_num.items():
                if k in target_uom_divisor:
                    target_uom_divisor = re.sub(k, str(v), target_uom_divisor)
            target_match_curr = re.search(currency_pattern,target_uom_divisor, re.I) # check if target uom divisor is a currency expression
            if target_match_curr:
                target_curr = (target_match_curr.groups()[0].strip() or target_match_curr.groups()[2].strip())
                try:
                    target_curr_num = int(target_match_curr.groups()[1].strip())
                except:
                    target_curr_num = 1

            if len(uom.split(' per '))>1: # if uom is an intensity uom and consistent with target_uom
                uom_base, uom_divisor = uom.split(' per ')
                for k, v in converter_big_num.items():
                    if k in uom_divisor:
                        uom_divisor = re.sub(k, str(v), uom_divisor)
                uom_match_curr = re.search(currency_pattern, uom_divisor, re.I)
                if (target_match_curr and not uom_match_curr) or (not target_match_curr and uom_match_curr):
                    print("Target/test UOM contains currency unit but test/target UOM doesn't contains currency. Invalid UOM conversion operation!")
                    return None, None
                elif target_match_curr and uom_match_curr:
                    uom_curr = (uom_match_curr.groups()[0].strip() or uom_match_curr.groups()[2].strip())
                    try:
                        uom_curr_num = int(uom_match_curr.groups()[1].strip())
                    except:
                        uom_curr_num = 1
                    multiplier_base, target_uom_base = uom_2_target_uom(uom_base, target_uom_base, metric_year, target_metric_density)
                    multiplier_uom_curr_divisor, target_uom_divisor = uom_2_target_uom(uom_curr, target_curr, metric_year, target_metric_density)
                    try:
                        multiplier_uom_divisor = multiplier_uom_curr_divisor * uom_curr_num / target_curr_num
                    except:
                        multiplier_uom_divisor = None
                else:
                    multiplier_base, target_uom_base = uom_2_target_uom(uom_base, target_uom_base, metric_year, target_metric_density)
                    multiplier_uom_divisor, target_uom_divisor = uom_2_target_uom(uom_divisor, target_uom_divisor, metric_year, target_metric_density)

                try:
                    multiplier = multiplier_base / multiplier_uom_divisor
                    target_uom = target_uom_base +' per ' + target_uom_divisor
                    return multiplier, target_uom
                except:
                    print("Either base UOM conversion multiplier or divisor UOM conversion multiplier is a NoneType. Invalid UOM conversion operation!")
                    return None, None
            else:
                # if uom cannot split with ' per ', that means uom is not an intensity uom and not able to match with target_uom, return none
                print("Test UOM is not an intensity UOM and not able to match with target_uom which is an intensity UOM")
                return None, None
        else:
            # replace large number representation by numerical value thats appears in uom
            for k, v in converter_big_num.items():
                if k in uom:
                    large_num = v
                    uom = re.sub(k, '', uom).strip()
                    break
            multiplier, target_uom = uom_2_target_uom(uom, target_uom, metric_year, target_metric_density)
            if 'large_num' in locals() and isinstance(multiplier,(float,int)):
                multiplier = multiplier * large_num
            return multiplier, target_uom
    else:
        return None, None