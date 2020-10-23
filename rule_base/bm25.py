import os
import sys
sys.path.append("../")
try:
    from load_icd_v3 import load_icd, load_train_icd, load_test_samples
except BaseException:
    from rule_base.load_icd_v3 import load_icd, load_train_icd, load_test_samples
from tqdm import tqdm
import pkuseg
import pickle
import numpy as np
import unicodedata
import string

def count_words(x, word_count):
    for word in x:
        if word in word_count.keys():
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def count_docs(x, word_count_doc):
    for word in set(x):
        if word in word_count_doc.keys():
            word_count_doc[word] += 1
        else:
            word_count_doc[word] = 1
    return word_count_doc

def string_clean_seg(x):
    s = unicodedata.normalize('NFKC', x)
    del_estr = string.punctuation + '、' #+ string.digits # ASCII 标点符号，数字
    replace = "\t"*len(del_estr)
    tran_tab = str.maketrans(del_estr, replace)
    s = s.translate(tran_tab)
    s = s.split('\t')
    return s

def get_word_tfidf(x_list, char_level = True):
    word_tfidf = {}
    word_count = {}
    word_count_doc = {}
    for x in tqdm(x_list):
        s = string_clean_seg(x)
        for sub_x in s:
            if char_level:
                x_l = list(sub_x)
            else:
                x_l = seg.cut(sub_x)
            word_count = count_words(x_l, word_count)
            word_count_doc = count_docs(x_l, word_count_doc)
    all_word_num = 0
    for word in list(word_count.keys()):
        all_word_num += word_count[word]
    for word in list(word_count_doc.keys()):
        if word_count_doc[word] > 1:
            word_tfidf[word] = np.sqrt(word_count[word] / all_word_num) * np.log2(8000 / (1 + word_count_doc[word]))
    return word_tfidf

def find_pairs(x, y, word_tfidf, co_exist, cleaned_y):
    for word in x:
        if word in word_tfidf.keys() and word_tfidf[word] > 0.001 and y in cleaned_y and len(set(word).intersection(set(y))) == 0 and y not in ['癌','恶性肿瘤']:
            key = word + '-' + y
            if key in co_exist.keys():
                co_exist[key] += word_tfidf[word]
            else:
                co_exist[key] = word_tfidf[word]
    return co_exist

def fill_co_exsit_dict(x_list, y_list, word_tfidf, char_level = True):
    co_exist = {}
    for i in tqdm(range(len(x_list))):
        if char_level:
            x_l = list(x_list[i])
        else:
            x_l = set(seg.cut(x_list[i]))
        for y in y_list[i]:
            # for y_word in seg.cut(y):
            co_exist = find_pairs(x_l, y, word_tfidf, co_exist)
    all_tfidf = {}
    for item in list(co_exist.keys()):
        tar_y = item.split('-')[1]
        if tar_y in all_tfidf.keys():
            all_tfidf[tar_y] += co_exist[item]
        else:
            all_tfidf[tar_y] = co_exist[item]
    for item in list(co_exist.keys()):
        co_exist[item] = co_exist[item] / all_tfidf[item.split('-')[1]]
    return co_exist

def fill_co_exsit_dict2(x_list, y_list, word_tfidf, char_level, cleaned_y):
    co_exist = {}
    for i in tqdm(range(len(x_list))):
        if char_level:
            x_l = list(x_list[i])
        else:
            x_l = set(seg.cut(x_list[i]))
        for y in y_list[i]:
            # for y_word in seg.cut(y):
            co_exist = find_pairs(x_l, y, word_tfidf, co_exist, cleaned_y)
    all_tfidf = 0
    for item in list(co_exist.keys()):
        all_tfidf += co_exist[item]
    for item in list(co_exist.keys()):
        co_exist[item] = co_exist[item] / all_tfidf
    return co_exist

def calculate_tfidf_simularity(x, y, co_exist):
    score = co_exist[x+'-'+y]
    return score

def dice(x, y):
    intersec = len(x.intersection(y))
    return max(2. * intersec / len(y), 2. * intersec / len(x))

def jaccard(x, y):
    unions = len(x.union(y))
    intersec = len(x.intersection(y))
    return intersec / unions

def set_find(x, y, metric = 'dice'):
    if metric == 'dice':
        if '[' in y:
            y_l = y.split('[')
            if y_l[0]:
                y = set(list(y_l[0]))
            else:
                y = set(list(y))
        else:
            y = set(list(y))
        score = dice(set(x), y)
    elif metric == 'jaccard':
        score = jaccard(set(x), set(y))
    return score

def set_match(x_list, 
            match_list_all, 
            match_list, 
            co_exist, 
            char_level = False,
            distance_metrics = 'dice',
            thre = 2000):
    pred_y = {}
    for i in tqdm(range(len(x_list))):
        #x_list[i] = ",".join(string_clean_seg(x_list[i]))
        tmp_x = ",".join(string_clean_seg(x_list[i]))
        now_y = {}
        if char_level:
            word_cut = list(set(tmp_x))
        else:
            word_cut = seg.cut(tmp_x)
        for idx, y in enumerate(match_list_all):
            tfidf_score = 0
            #dice_score = set_find(x_list[i], y, distance_metrics)
            dice_score = set_find(tmp_x, y, distance_metrics)
            now_y[y] = dice_score
            for word in word_cut:
                if word+'-'+y in co_exist.keys():
                    tfidf_score +=  co_exist[word +'-'+y]*thre
            # if np.log(tfidf_score+1) < 2:
            now_y[y] += tfidf_score
        pred_y[x_list[i]] = [a[0] for a in sorted(now_y.items(), key = lambda kv: -kv[1]) if a[1] > 0]
        # print(x_list[i])
        # print(seg_x[i])
        # print([a[1] for a in sorted(now_y.items(), key = lambda kv: -kv[1])][:30])
        # print([a[0] for a in sorted(now_y.items(), key = lambda kv: -kv[1])][:30])
        # input()
    return pred_y



if __name__ == '__main__':
    seg = pkuseg.pkuseg(model_name='medicine') 
    x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list= load_train_icd()
    x_list_test = load_test_samples()
    clean2standard, origin_y, standard2clean, cleaned_y= load_icd()

    # word_tfidf = get_word_tfidf(x_list, char_level = True)
    # input_x = x_list
    # input_y = y_list
    # co_exist = fill_co_exsit_dict2(input_x, input_y, word_tfidf, char_level = True)

    # tar_y_list = cleaned_y
    # # x_list = ['胃印戒细胞癌恶性肿瘤肿瘤 局部进展期,肺癌恶性肿瘤肿瘤术后 免疫力低 过敏性皮炎,上呼吸道感染 发热 咳嗽', '子宫颈中分化鳞癌恶性肿瘤肿瘤iib期放射治疗','多动症','耳后腮裂漏管']
    # result = set_match(x_list_test, cleaned_y, tar_y_list, co_exist, char_level = True ,distance_metrics = 'dice', thre =2000)
    # with open('../model/screen_test.pkl', 'wb') as f:
    #     pickle.dump(result, f)





#==================================================================================================

########### code for cross validation on training dataset, for searching hyperparams. ###########
    a = {}
    for i in range(0, 8):
        word_tfidf = get_word_tfidf(x_list, char_level = True)
        input_x = [item for item in x_list if item not in x_list[i*1000: i*1000+1000]]
        input_y = [y_clean_list[k] for k in range(len(y_clean_list)) if k < i*1000 or k >= i*1000+1000]
        co_exist = fill_co_exsit_dict2(input_x, input_y, word_tfidf, char_level = True)
        sort_freq = sorted(co_exist.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        tar_y_list = cleaned_y
        # x_list = ['胃印戒细胞癌恶性肿瘤肿瘤 局部进展期,肺癌恶性肿瘤肿瘤术后 免疫力低 过敏性皮炎,上呼吸道感染 发热 咳嗽', '子宫颈中分化鳞癌恶性肿瘤肿瘤iib期放射治疗','多动症','耳后腮裂漏管']
        result = set_match(x_list[i*1000: i*1000+1000], cleaned_y, tar_y_list, co_exist, char_level = True ,distance_metrics = 'dice', thre =2000)
        a.update(result)
    with open('../models/screen_train.pkl', 'wb') as f:
        pickle.dump(a, f)

#==================================================================================================

######### BM25 algorithm part, abandoned. ############

# import gensim.summarization.bm25 as BM25
# corpus = [list(set(y)) for y in cleaned_y]
# class_bm25 = BM25.BM25(corpus)
# pred_y = {}
# for x in tqdm(x_list):
#     scores = class_bm25.get_scores(list(set(x)))
#     now_y = {}
#     for i in range(len(cleaned_y)):
#         now_y[cleaned_y[i]] = scores[i]
#     now_y = [a[0] for a in sorted(now_y.items(), key = lambda kv: -kv[1])]
#     pred_y[x] = now_y[:300]
# with open('./result_dict/bm25.pkl', 'wb') as f:
#     pickle.dump(pred_y, f)
