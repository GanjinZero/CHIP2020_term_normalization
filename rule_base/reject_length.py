with open('../models/same_char_count_dict_tmp.txt') as f:
    lines = f.readlines()
x = [line.split("\t")[0] for line in lines]
y = [line.strip().split("\t")[1].split('##') for line in lines]
s = [float(line.strip().split("\t")[2]) for line in lines]


# reject s2
def inner_reject_pattern(y_list, s1, s2):
    f1 = False
    f2 = False
    for y in y_list:
        if s1 in y:
            f1 = True
        if s2 in y:
            f2 = True
    if f1 and f2:
        new_y_list = []
        for y in y_list:
            if s1 in y:
                new_y_list.append(y)
        return new_y_list
    else:
        return y_list

def inner_reject(x, y_list):
    y_list = inner_reject_pattern(y_list, "未特指的", "其他的")
    y_list = inner_reject_pattern(y_list, "未特指", "其他特指")
    y_list = inner_reject_pattern(y_list, "开放", "闭合")
    y_list = inner_reject_pattern(y_list, "完全", "部分")
    y_list = inner_reject_pattern(y_list, "开放", "闭合")
    y_list = inner_reject_pattern(y_list, "肿胀", "肿物")
    y_list = inner_reject_pattern(y_list, "肿胀", "水肿")
    return y_list

for i in range(len(x)):
    if len(y[i]) > 1 and s[i] < 1:
        y[i] = inner_reject(x[i], y[i])
    if s[i] <= 0.8:
        y[i] = []
    
dic = {}
for i in range(len(x)):
    if len(y[i]) > 0:
        dic[x[i]] = y[i]

import pickle
with open("../models/rejected_dic_tmp.pkl", "wb") as f:
    pickle.dump(dic, f)