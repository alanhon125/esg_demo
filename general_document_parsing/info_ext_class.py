import os
import pandas as pd
import re
import json


class text_info_ext:
    def __init__(self, text=None):
        self.text = text

        date_patterns = [r"(?<=\s)\d{4}", r"(?<=\s)\d{4}\s[A-Za-z]{3,9}\s\d{1,2}", r"(?<=\s)\d{1,2}\s[A-Za-z]{3,9}\s\d{4}", r"(?<=\s)\d{4}\s[A-Za-z]{3}\.\s\d{1,2}",
                         r"(?<=\s)\d{1,2}\s[A-Za-z]{3}\.\s\d{4}", r"(?<=\s)\d{4}-\d{1,2}-\d{1,2}", r"(?<=\s)\d{1,2}-\d{1,2}-\d{4}", r"(?<=\s)\d{4}\/\d{1,2}\/\d{1,2}", r"(?<=\s)\d{1,2}\/\d{1,2}\/\d{4}"]

        """ numbers_patterns """
        number_patterns = [r"\-?\d+", r"\-?\d+%", r"\-?\d{1,3},\d{3}", r"\-?\d{1,3},\d{3}%", r"\-?\d{1,3},\d{3},\d{3}", r"\-?\d{1,3},\d{3},\d{3}%", r"\-?\d{1,3},\d{3},\d{3},\d{3}", r"\-?\d{1,3},\d{3},\d{3},\d{3}%", r"\-?\d+\.\d+",
                           r"\-?\d+\.\d+%", r"\-?\d{1,3},\d{3}\.\d+", r"\-?\d{1,3},\d{3}\.\d+%", r"\-?\d{1,3},\d{3},\d{3}\.\d+", r"\-?\d{1,3},\d{3},\d{3}\.\d+%", r"\-?\d{1,3},\d{3},\d{3},\d{3}\.\d+", r"\-?\d{1,3},\d{3},\d{3},\d{3}\.\d+%"]

        """ keywords """
        keyword_patterns = [r"GHG", r"Greenhouse gases", r"Greenhouse gases [(]GHG[)]", r"GHG emissions?", r"GHG scope 1 emissions?", r"GHG scope 2 emmisions?", r"scope 1", r"scope 2", r"scope 3", r"intensity", r"emission intensity", r"GHG emission intensity", r"green house gas", r"non-hazardous wastes?", r"hazardous wastes?", r"hazardous and non-hazardous wastes?", r"other non-hazardous wastes?", r"total carbon emissions?", r"carbon emissions?", r"sulphur oxides?", r"sulphur oxides? [(]SOx[)]", r"nitrogen oxides?", r"nitrogen oxides? [(]NOx[)]", r"particulate matters?", r"particulate matters? [(]PM[)]", r"sulfur dioxides?", r"nitrogen dioxides?", r"chemical oxygen demands?", r"chemical oxygen demands? [(]COD[)]", r"carbon dioxide equivalents?",
                            r"tCO2e", r"carbon dioxide equivalents? [(]tCO2e[)]", r"carbon dioxides?", r"CO2", r"carbon dioxides? [(]CO2[)]", r"methanes?", r"methanes? [(]CH4[)]", r"methanol", r"nitrous oxides?", r"nitrous oxides? [(]N2O[)]", r"ammonia nitrogen", r"ammonia-nitrogen", r"HFC", r"PFC", r"Volatile organic compounds?", r"O3", r"Ozone [(]O3[)]", r"PM10", r"Coarse particles? [(]PM10[)]", r"Dust [(]total measurable particles?[)]", r"PM2.5", r"Small particles? dust [(]PM2.5[)]", r"Lead [(]Pb[)]", r"C20H12", r"Benzo[(]a[)] pyrene [(]C20H12[)]", r"coal gangues?", r"coal fly ash", r"cinders?", r"chemical wastes?", r"muds?", r"rocks?", r"industrial water", r"mine water", r"emissions", r"oily sludges?", r"oily wastes?", r"industry wastes?", r"office wastes?", r"NOx emissions?"]

        """ measurements """
        measurement_patterns = [r"Mwh", r"Megawatt hours", r"Megawatt hours [(]MWh[)]", r"tonnes?", r"tons?",
                                r"metric tonnes?", r"metric tons?", r"kgs?", r"million tonnes?", r"million tons?", r"cubic meters?"]
        intensity_patterns = [r"kg/m2", r"kg/employee",
                              r"tCO2e/m2", r"tCO2e/employee"]
        currency_patterns = [r"RMB", r"USD", r"HKD"]

        # """ Verbs, conjunctions, prepositions """
        # # r"(?<=\s)WORD(?=\s)" to match VCP keyword exactly, not as a part of a long word
        # VCP_patterns = [r"(?<=\s)of(?=\s)", r"(?<=\s)and(?=\s)", r"(?<=\s)or(?=\s)", r"(?<=\s)am(?=\s)",
        #                 r"(?<=\s)is(?=\s)", r"(?<=\s)are(?=\s)", r"(?<=\s)was(?=\s)", r"(?<=\s)were(?=\s)"]

        """ all patterns together """
        # patterns_list = keyword_patterns + number_patterns + date_patterns + measurement_patterns + intensity_patterns + VCP_patterns
        patterns_list = keyword_patterns + number_patterns + date_patterns + \
            measurement_patterns + intensity_patterns + currency_patterns

        self.text_data, self.matchlist, self.result = self.text_information_extraction(
            text, number_patterns, keyword_patterns, measurement_patterns, patterns_list)

    def text_information_extraction(self, text, number_patterns, keyword_patterns, measurement_patterns, patterns_list):
        if len(text) == 0:
            print("Please give a valid text.")
            return None

        results_df = pd.DataFrame(columns=['number', 'num_start_pos', 'num_end_pos', 'measurement',
                                           'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])

        match_list = self.find_all_patterns(patterns_list, text)
        matchlist_norddt = self.matchlist_noredundant(match_list)

        rule_list = self.rule1_number_meas(
            number_patterns, measurement_patterns, keyword_patterns, matchlist_norddt)
        temp_df = self.apply_rule1(rule_list, matchlist_norddt)

        if (temp_df is not None):
            results_df = pd.concat(
                [results_df, temp_df], ignore_index=True)

        rule_list = self.rule2_rep3time_num_meas(
            number_patterns, measurement_patterns, keyword_patterns, matchlist_norddt)

        temp_df = self.apply_rule2(rule_list, matchlist_norddt)
        if (temp_df is not None):
            results_df = pd.concat(
                [results_df, temp_df], ignore_index=True)

        return text, matchlist_norddt, results_df

    def find_all_patterns(self, patterns, text):
        """find all patterns in pattern-list for given text"""
        match_list = []
        for pattern in patterns:
            text_temp = text
            text_lenth = len(text_temp)
            start_pos = 0
            end_pos = 0
            index_shift = 0
            while text_lenth > 0:
                # not case sensitive
                result = re.search(pattern, text_temp, re.IGNORECASE)
                if result is None:
                    break
                start_pos = result.start()
                end_pos = result.end() - 1
                match_list.append(
                    [result[0].strip(), start_pos + index_shift, end_pos + index_shift])
                index_shift += end_pos + 1
                text_temp = text_temp[end_pos + 1:]
                text_lenth = len(text_temp)

        return match_list

    def matchlist_noredundant(self, match_list):
        results = []
        length = len(match_list)
        if len(match_list) == 0:
            return None
        if length == 1:
            return self.output_DataFrame(match_list)
        else:
            for i in range(length):
                for j in range(length):
                    if (i != j) and (self.is_first_redundant(match_list[i], match_list[j])):
                        break
                    if (j == length - 1) and (match_list[i] not in results):
                        results.append(match_list[i])

        return self.output_DataFrame(results)

    def is_first_redundant(self, item1, item2):
        return (item1[1] >= item2[1]) and (item1[2] < item2[2]) or (item1[1] > item2[1]) and (item1[2] <= item2[2])

    def output_DataFrame(self, match_list):
        if len(match_list) == 0:
            return None
        else:
            results_df = pd.DataFrame(match_list)
            results_df.columns = ['keyword', 'start_pos', 'end_pos']
            results_df.sort_values(
                by='start_pos', ascending=True, inplace=True, ignore_index=True)
            return results_df

    def is_alnumber(self, number_patterns, test_string):
        """ check whether the string is purely numbers """
        result = self.find_all_patterns(number_patterns, test_string)
        if len(result) == 0:
            return False
        else:
            return True

    def is_keyword_in_patterns(self, patterns, keyword):
        """ #check the type of keyword """
        for pattern in patterns:
            if re.search(pattern, keyword, re.IGNORECASE) is not None:
                return True
        return False

    def rule1_number_meas(self, number_patterns, measurement_patterns, keyword_patterns, matchlist_df):
        if matchlist_df is None:
            return []

        if matchlist_df.shape[0] <= 2:
            return []

        result = []
        for i in range(matchlist_df.shape[0]-2):
            item1 = matchlist_df.iloc[i]
            item2 = matchlist_df.iloc[i+1]
            item3 = matchlist_df.iloc[i+2]

            if (abs(item2['start_pos'] - item1['end_pos']) == 2) and (abs(item3['start_pos'] - item2['end_pos']) == 5) and (self.is_keyword_in_patterns(number_patterns, item1[0])) and (self.is_keyword_in_patterns(measurement_patterns, item2[0])) and (self.is_keyword_in_patterns(keyword_patterns, item3[0])):
                result.append(i)

        return result

    def apply_rule1(self, rule_list, matchlist_df):

        if len(rule_list) == 0:
            return None

        matched_df = []

        for index in rule_list:
            item1 = matchlist_df.iloc[index]
            item2 = matchlist_df.iloc[index+1]
            item3 = matchlist_df.iloc[index+2]
            matched_item = [item1[0], item1[1], item1[2], item2[0],
                            item2[1], item2[2], item3[0], item3[1], item3[2]]
            matched_df.append(matched_item)

        matched_df = pd.DataFrame(matched_df, columns=['number', 'num_start_pos', 'num_end_pos',
                                                       'measurement', 'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])

        return matched_df

    # rule2: number + measurement + "of" + keywords

    def rule2_rep3time_num_meas(self, number_patterns, measurement_patterns, keyword_patterns, matchlist_df):
        if matchlist_df is None:
            return []

        if matchlist_df.shape[0] <= 8:
            return []

        result = []
        for i in range(matchlist_df.shape[0]-8):
            item1 = matchlist_df.iloc[i]
            item2 = matchlist_df.iloc[i+1]
            item3 = matchlist_df.iloc[i+2]
            item4 = matchlist_df.iloc[i+3]
            item5 = matchlist_df.iloc[i+4]
            item6 = matchlist_df.iloc[i+5]
            item7 = matchlist_df.iloc[i+6]
            item8 = matchlist_df.iloc[i+7]
            item9 = matchlist_df.iloc[i+8]

            if (abs(item2['start_pos'] - item1['end_pos']) == 2) and (abs(item4['start_pos'] - item3['end_pos']) == 2) and (abs(item6['start_pos'] - item5['end_pos']) == 2) and (abs(item7['start_pos'] - item6['end_pos']) == 5) and (self.is_keyword_in_patterns(number_patterns, item1[0])) and (self.is_keyword_in_patterns(measurement_patterns, item2[0])) and \
                (self.is_keyword_in_patterns(number_patterns, item3[0])) and (self.is_keyword_in_patterns(measurement_patterns, item4[0])) and \
                (self.is_keyword_in_patterns(number_patterns, item5[0])) and (self.is_keyword_in_patterns(measurement_patterns, item6[0])) and \
                    (self.is_keyword_in_patterns(keyword_patterns, item7[0])) and (self.is_keyword_in_patterns(keyword_patterns, item8[0])) and (self.is_keyword_in_patterns(keyword_patterns, item9[0])):
                result.append(i)
                print(i)

        return result

    def apply_rule2(self, rule_list, matchlist_df):

        if len(rule_list) == 0:
            return None

        matched_df = []

        for index in rule_list:
            item1 = matchlist_df.iloc[index]
            item2 = matchlist_df.iloc[index+1]
            item3 = matchlist_df.iloc[index+2]
            item4 = matchlist_df.iloc[index+3]
            item5 = matchlist_df.iloc[index+4]
            item6 = matchlist_df.iloc[index+5]
            item7 = matchlist_df.iloc[index+6]
            item8 = matchlist_df.iloc[index+7]
            item9 = matchlist_df.iloc[index+8]

            matched_item1 = [item1[0], item1[1], item1[2], item2[0],
                             item2[1], item2[2], item7[0], item7[1], item7[2]]
            matched_item2 = [item3[0], item3[1], item3[2], item4[0],
                             item4[1], item4[2], item8[0], item8[1], item8[2]]
            matched_item3 = [item5[0], item5[1], item5[2], item6[0],
                             item6[1], item6[2], item9[0], item9[1], item9[2]]

            matched_df.append(matched_item1)
            matched_df.append(matched_item2)
            matched_df.append(matched_item3)

        matched_df = pd.DataFrame(matched_df, columns=['number', 'num_start_pos', 'num_end_pos',
                                                       'measurement', 'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])

        return matched_df


def has_key(text_dict, keyword):
    """if give dict has the key of kewword"""
    return keyword in text_dict.keys()


def get_text_from_json(json_file_path):
    """get pure texts from generated json file

    Args:
        json_file_path (json file): generated json file from doc parsing
    Output: 
        List of pure texts from generated json file
    """
    pure_texts = []

    with open(json_file_path, 'r') as f:
        text_data = json.load(f)

    for item in text_data['content']:
        if has_key(item, 'paragraph'):
            pure_texts.append(re.sub('\s', ' ', item['paragraph']))
        if has_key(item, 'child_content'):
            if item['child_content'] is not None:
                for child_item in item['child_content']:
                    if has_key(child_item, 'paragraph'):
                        pure_texts.append(
                            re.sub('\s', ' ', child_item['paragraph']))

    return pure_texts


def extract_text_info(texts):
    """extract information from list of texts

    Args:
        texts (list of string)
    Output: 
        DataFrame of extracted information from texts
    """
    # results_df = pd.DataFrame(columns=['number', 'num_start_pos', 'num_end_pos', 'measurement',
    #                                    'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])

    final_results = []

    for text in texts:
        results = dict()
        results['text'] = text
        results_df = pd.DataFrame(columns=['number', 'num_start_pos', 'num_end_pos', 'measurement',
                                           'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])
        model = text_info_ext(text)
        if model.result.shape[0] > 0:
            results_df = pd.concat(
                [results_df, model.result], ignore_index=True)

            results['info'] = results_df

            final_results.append(results)

    final_results = pd.DataFrame(final_results)

    return final_results


# def extract_text_info(texts):
#     """extract information from list of texts

#     Args:
#         texts (list of string)
#     Output:
#         DataFrame of extracted information from texts
#     """
#     results_df = pd.DataFrame(columns=['number', 'num_start_pos', 'num_end_pos', 'measurement',
#                                        'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])

#     for text in texts:
#         model = text_info_ext(text)
#         if model.result.shape[0] > 0:
#             results_df = pd.concat(
#                 [results_df, model.result], ignore_index=True)

#     return results_df


def extract_text_info_all(texts):
    """extract information from list of texts

    Args:
        texts (list of string)
    Output: 
        DataFrame of extracted information from texts
    """
    results_df = pd.DataFrame(columns=['number', 'num_start_pos', 'num_end_pos', 'measurement',
                                       'meas_start_pos', 'meas_end_pos', 'keyword', 'key_start_pos', 'key_end_pos'])

    results_all_df = pd.DataFrame(
        columns=['keyword', 'start_pos', 'end_pos'])

    for text in texts:
        model = text_info_ext(text)
        if model.matchlist is not None:
            results_all_df = pd.concat(
                [results_all_df, model.matchlist], ignore_index=True)
        if model.result.shape[0] > 0:
            results_df = pd.concat(
                [results_df, model.result], ignore_index=True)

    return results_df, results_all_df


if __name__ == "__main__":

    # text = """During the year, the Group’s total carbon emissions were 38,124 metric tons of carbon dioxide equivalent, mainly from the purchased electricity, accounting for 68% of the total carbon emissions. Compared with last year’s performance, the Group’s total GHG emissions increased 87%, which was mainly generated by the increasing work load of construction sites. The GHG emission intensity by the number of employees is 11.1, while the intensity by RMB 1,000,000 is 10.5.

    # Greenhouse gases (GHG) were generated directly from the consumption of stationary and mobile fuel, and indirectly from the consumption of purchased electricity and steam, processing of freshwater and sewage, landfilling of waste papers, and air travels taken by employees for work purposes. During the Reporting Period, the Group emitted 9,227.41 tonnes of carbon dioxide equivalent (tCO2e) of GHG (mainly carbon dioxide, methane and nitrous oxide), with an intensity of 0.002 tCO2e/m2, and 9.73 tCO2e/employee.

    # Motor vehicles owned by the Group are mainly for providing transportation for the employees in shipyards and for the use of senior management. Among all the company vehicles, 10 are driven by unleaded petroleum while only 1 of the cars is driven by diesel. During the process of combustion of unleaded petrol and diesel, 7.32 kg, 0.15 kg and 0.54 kg of nitrogen oxides (“NOx”), sulphur oxides (“SOx”) and particulate matters (“PM”) are produced respectively.

    # During the Reporting Period, the Group emitted a total of 2.31 kg sulphur oxides (SOx), 15.29 kg of nitrogen oxides (NOx), and 1.48 kg of particulate matters (PM).

    # For Hong Kong, according to Hong Kong Electric Company, with 1 kilowatt hour (“kWh”) of power generated through combustion of fuel, 0.81 kg of CO2 is produced. For Russia, according to Carbon Footprint Country Specific Electricity Grid Greenhouse Gas Emission Factors, with 1 kWh of power generated through combustion of fuel, 0.325 kg of CO2 is produced. For Korea,  the consumption of electricity is limited and the relevant fee is included in the monthly rental fee, therefore, it does not constitute a focus in the ESG Report.
    # """

    file_path = 'data/docparse_json/巨濤海洋石油服務_Environmental,SocialandGovernanceReport2020.json'

    texts = get_text_from_json(file_path)

    results = extract_text_info(texts)

    print(results)
    print(results.to_json(orient='records'))
