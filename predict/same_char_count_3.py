# -*- coding:utf-8 -*-

import sys
sys.path.append("../rule_base")
from load_icd_v3 import load_icd, load_train_icd
from load_icd_v3 import load_test_samples as load_test_icd
import pickle
from tqdm import tqdm, trange
import numpy as np
from rule_match import match_yun
from postprocess import reject



def score(x, y):
    #if reject(x, y):
    #    return 0.
    #if y in match_yun(x):
    #    return 1.
    '''
    l = len(y)
    if l == 1:
        return 0.
    x_body = set(extract_body(x))
    y_body = set(extract_body(y))
    s_body = 0
    for b in y_body:
        if b in x_body:
            s_body += 1
    order = []
    for cy in y:
        if cy in x:
            order.append(x.find(cy))
    s = len(order)
    continuous = sum([order[i+1] == order[i] + 1 for i in range(s - 1)])
    asc = sum([order[j] > order[i] for i in range(s - 1) for j in range(i + 1, s) ])
    return s / l / 2 + continuous / (l - 1) / 3 + 2 * asc / l / (l - 1) / 3 + s_body / l

    '''
    set_x = x
    set_y = y
    if len(set_y) < 2:
        return 0.
    s = len(set_x.intersection(set_y))
    if s == len(set_y):
        return 1.
    if s == len(set_x):
        return 1.
    total = min(len(set_x), len(set_y))
    return s / total
    '''
    if len(y) > 1:
        set_x = set([x[i:i+2] for i in range(0, len(x) - 1)])
        set_y = set([y[i:i+2] for i in range(0, len(y) - 1)])
        s += len(set_x.intersection(set_y))
        total += len(set_y)

    if len(y) > 2:
        set_x = set([x[i:i+3] for i in range(0, len(x) - 2)])
        set_y = set([y[i:i+3] for i in range(0, len(y) - 2)])
        s += len(set_x.intersection(set_y))
        total += len(set_y)
    
    if len(y) > 3:
        set_x = set([x[i:i+4] for i in range(0, len(x) - 3)])
        set_y = set([y[i:i+4] for i in range(0, len(y) - 3)])
        s += len(set_x.intersection(set_y))
        total += len(set_y)
    '''


def find_same_char_max(x, cleaned_y, clean2standard, th, set_x, set_y):
    max_score = 0.
    max_y = []
    match_yun_x = match_yun(x)
    for i in range(len(cleaned_y)):
        y = cleaned_y[i]
        if y in match_yun_x:
            s = 1.
        else:
            s = score(set_x, set_y[i])
        if s > max_score:
            if not reject(x, y):
                max_score = s
                max_y = [clean2standard[y]]
        elif s == max_score:
            if not reject(x, y):
                max_y.append(clean2standard[y])
    return max_y, max_score

'''
def find_same_char_max(x, cleaned_y, clean2standard, th, set_x, set_y):
    max_score = 0.
    max_y = ""
    for i in range(len(cleaned_y)):
        if x in match_yun(cleaned_y[i]):
            s = 1.
        else:
            s = score(set_x, set_y[i])
        if s > max_score and s > th:
            if not reject(x, cleaned_y[i]):
                max_score = s
                max_y = cleaned_y[i]
    if max_y == "":
        return "", max_score
    else:
        return clean2standard[max_y], max_score
'''
#new_x_list, _, _, _, new_y_list = load_train_icd()
clean2standard, origin_y, standard2clean, cleaned_y = load_icd()
new_x_list = load_test_icd()

target = []
sc = []
#s_t = []


#new_x_list = new_x_list[:100]
#new_y_list = new_y_list[7000:]

'''
with open("1024_150_0.001_5_word2vec_5_300_Dice_tfidfjson_90_0.8_(-1, 3)_2_none.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    new_x_list.append(line.split('\t')[0])
    new_y_list.append(line.split('\t')[2].strip().split('##'))
    if '' in new_y_list[-1]:
        new_y_list[-1].remove('')
'''

th = 0.8
set_x = [set(xx) for xx in new_x_list]
set_y = [set(y) for y in cleaned_y]
for i in trange(len(new_x_list)):
    x = new_x_list[i]
    t, s = find_same_char_max(x, cleaned_y, clean2standard, th, set_x[i], set_y)
    #t, s = find_same_char_max(x, cleaned_y, clean2standard, th, set_x[i], set_y)
    target.append(t)
    sc.append(s)
    #s_t.append(s_true)

'''
best_th = 0.
best_acc = 0.
best_dic = None
for th in np.arange(0, 1.1, 0.05):
    f = open(f"same_char_count_{th}.txt", "w+")
    same_char_count_dict = {}
    count = 0
    exact_count = 0
    pred = []
    for i in range(len(new_x_list)):
        x = new_x_list[i]
        right = False
        if sc[i] >= th:
            same_char_count_dict[x] = target[i]
            right = set(same_char_count_dict[x]) == set(new_y_list[i])
            pred.append(len(target[i]))
        else:
            right = len(new_y_list[i]) == 0
            pred.append(0)
        f.write('\t'.join([x, '##'.join(target[i]), str(sc[i]), str(right), '##'.join(new_y_list[i]), str(s_t[i])]) + '\n')
        if right:
            count += 1
    acc = count / len(new_x_list)
    print(th, np.bincount(pred), acc)
    if acc > best_acc:
        best_th = th
        best_acc = acc
        best_dic = same_char_count_dict
    f.close()
print(best_th, best_acc)
'''

f = open(f"../models/same_char_count_dict_tmp.txt", "w+")
same_char_count_dict = {}
for i in range(len(new_x_list)):
    x = new_x_list[i]
    if sc[i] > th:
        same_char_count_dict[x] = (target[i], sc[i])
    f.write('\t'.join([x, '##'.join(target[i]), str(sc[i]) + '\n']))
f.close()

#with open(f"../models/same_char_count_dict.pkl", "wb") as f:
#    pickle.dump(same_char_count_dict, f)

