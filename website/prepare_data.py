import sys
sys.path.append("../")
import rule_base.load_icd_v3 as load_icd_v3
from rule_base.load_icd_v3 import load_train_icd, load_icd_code, clean
from rule_base.bm25 import get_word_tfidf, fill_co_exsit_dict2, string_clean_seg, set_find



def init():
    x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list = load_train_icd()
    clean2standard, origin_y, standard2clean, cleaned_y, clean2code = load_icd_code()
    return x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list, \
           clean2standard, origin_y, standard2clean, cleaned_y, clean2code

"""
def calculate_co_exist(x_list, y_clean_list, cleaned_y):
    word_tfidf = get_word_tfidf(x_list, char_level=True)
    co_exist = fill_co_exsit_dict2(x_list, y_clean_list, word_tfidf, char_level=True, cleaned_y=cleaned_y)
    sort_freq = sorted(co_exist.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    return co_exist
"""
    
x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list, \
    clean2standard, origin_y, standard2clean, cleaned_y, clean2code = init()

import json
with open("data/clean2code.json", "w") as f: json.dump(clean2code, f)
#co_exist = calculate_co_exist(x_list, y_clean_list, cleaned_y)
#import ipdb; ipdb.set_trace()