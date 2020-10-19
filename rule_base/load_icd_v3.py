import pandas as pd
import os
import re
import json
from tqdm import tqdm
import pickle

i_to_num_dict = {'i':'1', 'ii':'2', 'iii':'3', 'iv':'4', 'v':'5', 'vi':'6', 'vii':'7', 'viii':'8'}

def match_itomun(substring):
    abbr = re.search('^v?i+v?', substring.groupdict()['pat'])
    if not abbr:
        abbr = re.search('v?i+v?$', substring.groupdict()['pat'])
    if not abbr:
        return substring.group()
    else:
        abbr = abbr.group()
        matched = re.sub(abbr, i_to_num_dict[abbr], substring.groupdict()['pat'])
        return matched

def i_to_num(string):
    if 'i' in string:
        string = re.sub('(?P<pat>[a-zA-Z]+)', match_itomun, string)
    return string

digit_map = {"Ⅳ":"iv", "Ⅲ":"iii", "Ⅱ":"ii", "Ⅰ":"i", "一":"1", "二":"2", "三":"3", "四":"4", "五":"5", "六":"6"}
def clean_digit(string):
    # Ⅳ Ⅲ Ⅱ Ⅰ
    # IV III II I
    # 4 3 2 1
    # 四 三 二 一
    new_string = ""
    for ch in string:
        if ch.upper() in digit_map:
            new_string = new_string + digit_map[ch.upper()]
        else:
            new_string = new_string + ch
    return new_string

greek_lower = [chr(ch) for ch in range(945, 970) if ch != 962]
greek_upper = [chr(ch) for ch in range(913, 937) if ch != 930]
greek_englist = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
                 "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
greek_map = {ch:greek_englist[idx % 24] for idx, ch in enumerate(greek_lower + greek_upper)}
def clean_greek(string):
    new_string = ""
    for ch in string:
        if ch in greek_map:
            new_string = new_string + greek_map[ch]
        else:
            new_string = new_string + ch
    return new_string

prefix_suffix_src = ["部位未特指的", "未特指的", "原因不明的", "意图不确定的", "不可归类在他处", "其他特指的疾患"]
prefix_suffix_tgt = ["部未指", "未指", "不明", "意不", "不归他", "他特指"]
def clean_prefix_suffix(string):
    for idx, replace_str in enumerate(prefix_suffix_src):
        string = string.replace(replace_str, prefix_suffix_tgt[idx])
    return string

try:
    with open('./other_map.json', 'r') as f:
        other_map = json.load(f)
except BaseException:
    with open('../data/other_map.json', 'r') as f:
        other_map = json.load(f)

def match(substring):
    abbr = re.search('[a-z]+', substring.groupdict()['pat']).group()
    matched = re.sub(abbr, other_map[abbr], substring.groupdict()['pat'])
    return matched

def clean_other(string):
    # oa
    # "＋"="+"
    # aoux not replace ou
    for item in list(other_map.keys()):
        if item == "＋":
            string = re.sub(item, other_map[item], ' '+string+' ')
        else:
            string = re.sub('(?P<pat>[^a-zA-Z]'+item+'[^a-zA-Z])', match, ' '+string+' ')
    return string.strip(' ')

def clean_index(string):
    # 1. 2.
    new_string = ""
    idx = 0
    while idx < len(string):
        ch = string[idx]
        if "0" <= ch <= "9" and idx < len(string) - 1 and string[idx + 1] == ".":
            new_string += " "
            idx += 1
        else:
            new_string += ch
        idx += 1
    return new_string

def clean(string):
    string = string.replace("\"", " ").lower()
    string = clean_index(string)
    string = clean_prefix_suffix(string)
    string = clean_greek(string)
    string = clean_digit(string)
    string = clean_other(string)
    string = i_to_num(string)
    string = clean_other(string)
    return string.lower()

prefix_suffix_src_x = ["恶性","癌", "慢支", "化疗", "皮肤", "胃口", "节育器",
                        "左甲","右甲","腮裂","白内障","小便","停经","积血"]
prefix_suffix_tgt_x = ["恶性肿瘤","癌恶性肿瘤","慢性支气管炎","化学治疗","皮肤和皮下组织", "食欲","避孕环",
                        "左甲状腺","右甲状腺","鳃裂","白内障眼","尿","孕","积血肿"]
def extend_x(string):
    for idx, replace_str in enumerate(prefix_suffix_src_x):
        string = string.replace(replace_str, prefix_suffix_tgt_x[idx])
    return string

def load_icd():
    if os.path.exists("ICD_10v601.csv"):
        df = pd.read_csv("ICD_10v601.csv", header=None)
    else:
        df = pd.read_csv("../data/ICD_10v601.csv", header=None)
    origin_y = set(df[1].tolist())
    clean2standard = {}
    for d in tqdm(origin_y):
        if clean(d) in clean2standard:
            print(clean(d), d)
        clean2standard[clean(d)] = d
    standard2clean = {v:k for k, v in clean2standard.items()}
    cleaned_y = list(clean2standard.keys())
    
    return clean2standard, origin_y, standard2clean, cleaned_y

def load_icd_code():
    if os.path.exists("ICD_10v601.csv"):
        df = pd.read_csv("ICD_10v601.csv", header=None)
    else:
        df = pd.read_csv("../data/ICD_10v601.csv", header=None)

    code = df[0].tolist()
    y_list = df[1].tolist()

    standard2code = {}
    for idx, y in enumerate(y_list):
        standard2code[y] = code[idx]

    clean2code = {}

    origin_y = set(df[1].tolist())
    clean2standard = {}
    for d in origin_y:
        clean_d = clean(d)
        if clean_d in clean2standard:
            print(clean_d, d)
        clean2standard[clean_d] = d
        clean2code[clean_d] = standard2code[d]

    standard2clean = {v:k for k, v in clean2standard.items()}
    cleaned_y = list(clean2standard.keys())
    return clean2standard, origin_y, standard2clean, cleaned_y, clean2code

def load_test_samples():
    if os.path.exists("test.txt"):
        train_path = "test.txt"
    else:
        train_path = "../data/test.txt"
    with open(train_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    x_list = []
    for line in lines:
        l = line.strip('\n').strip()
        x_list.append(extend_x(clean(l)))
    return x_list


def load_train_icd():
    clean2standard, origin_y, standard2clean, cleaned_y= load_icd()
    if os.path.exists("train.txt"):
        train_path = "train.txt"
    else:
        train_path = "../data/train.txt"
    with open(train_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    x_list = []
    y_list = []
    y_clean_list = []
    y_icd10_list = []
    y_clean_icd10_list = []
    for line in lines:
        l = line.strip().split("\t")
        x_list.append(extend_x(clean(l[0])))
        y_list.append(l[1].split("##"))
        y_clean_list.append([clean(y) for y in y_list[-1]])
        y_icd10_list.append([y for y in y_list[-1] if y in origin_y])
        #import ipdb; ipdb.set_trace()
        y_clean_icd10_list.append([standard2clean[y] for y in y_icd10_list[-1]])

    return x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list

# def reject(x, y):
#     pattern_y = re.findall('[1-8][期度型级]', y)
#     pattern_x = re.findall('[1-8][期度型级]', x)
#     raw_num =  re.findall('[^0-9a-z][1-8][^0-9期度型级周年日点]', x)
#     for item in raw_num:

#     print(pattern_x)
#     for item in pattern_y:
#         if item in pattern_x or item[0] in pattern_x:
#             continue
#         else:
#             print(item[0])
#             return True
#     return False

if __name__ == "__main__":
    # with open('./dice_test_sample_new.pkl', 'rb') as f:
    #     aa = pickle.load(f)
    # print(aa['朗格汉斯组织细胞增生症多系统型累及肺骨骼'][:30])

    # a = '哈vi手viniii-ii机的哈ciniii萨克NYHAIII-IV多久啊viabc刷空间'
    a = '"右肺上叶腺癌VINII III期,放疗，食管炎,便"'
    print(clean(a))
    print(a)

    # x = '客户端卡iicns多久cin[vi]啊大cin很快收iii到ciniv贺卡'
    # x= '子宫颈上皮内瘤变III级[CINIII级]'.lower()
    # y = '好的快结婚了卡升级iv卡号分开多久'
    # print(i_to_num(x))


    # x = '局部II鳞状上I皮呈cinii级并累及腺体'
    # y = '子宫颈上皮内I瘤变III级[CINIII级]','子宫颈II级上皮瘤样I级病变[CIN]II级'
    # for i in y:
    #     print(clean(x), clean(i))
    #     print(reject(clean(x), clean(i)))
    # x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list = load_train_icd()
    # clean2standard, origin_y, standard2clean, cleaned_y = load_icd()
    # print(clean(a) in x_list)
    # b = '神经纤维瘤病[vonrecklinghausen病]'
    # c = '神经纤维瘤病'
    # print(b in cleaned_y)
    # print(c in cleaned_y)