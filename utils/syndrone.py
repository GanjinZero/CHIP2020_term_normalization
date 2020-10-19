import re
import copy
import pickle
import numpy as np
import sys
sys.path.append('../')
from rule_base.load_icd_v3 import load_icd, load_test_samples
def syndrome(x_list, y_list, standard_icd):
    for i in range(len(x_list)):
        if '综合征' in x_list[i]:
            eng_x = re.findall('[a-zA-Z- ]{3,}', x_list[i])
            if len(y_list[i]) > 3:
                out = []
                for item_y in y_list[i]:
                    if '综合征' in item_y:
                        eng_y = re.findall('[a-zA-Z- ]{3,}', item_y)
                        if bool(eng_x) and bool(eng_y):
                            # intersec_lower = set(eng_x[0].lower()).intersection(set(eng_y[0].lower()))
                            intersec = set(eng_x[0].replace('S', '')).intersection(set(eng_y[0]))
                            dice = max(len(intersec)/len(set(eng_y[0])), len(intersec)/len(set(eng_x[0])))
                            # dice_lower = max(len(intersec_lower)/len(set(eng_y[0].lower())), len(intersec_lower)/len(set(eng_x[0].lower())))
                            if (dice == 1 and len(intersec)>=2) or (eng_x[0].lower() == eng_y[0].lower()):
                                out.append(item_y)
                        else:
                            out.append(item_y)
                    else:
                        out.append(item_y)
                y_list[i] = out

            if not y_list[i] or y_list[i] == ['']:
                for item in standard_icd:
                    if '综合征' in item:
                        eng = re.findall('[a-zA-Z- ]{3,}', item)
                        if eng_x:
                            if eng:
                                if len(eng_x[0]) < 4:
                                    intersec = set(eng_x[0].replace('S', '')).intersection(set(eng[0]))
                                    dice = max(len(intersec)/len(set(eng[0])), len(intersec)/len(set(eng_x[0].replace('S', ''))))
                                
                                else:
                                    intersec = set(eng_x[0]).intersection(set(eng[0]))
                                    dice = max(len(intersec)/len(set(eng[0])), len(intersec)/len(set(eng_x[0])))
                                
                                if (dice == 1 and len(intersec)>=2) or (eng_x[0].replace('-','').lower() == eng[0].replace('-','').lower()):
                                    y_list[i].append(item)
                        else:
                            if eng:
                                x_set = set(x_list[i].replace('综合征',''))
                                y_set = set(item.replace('综合征','').replace(eng[0],''))
                                intersec = x_set.intersection(y_set)
                            else:
                                x_set = set(x_list[i].replace('综合征',''))
                                y_set = set(item.replace('综合征',''))
                                intersec = x_set.intersection(y_set)
                            if len(intersec) > 0:
                                dice = max(len(intersec)/len(x_set), len(intersec)/len(y_set))
                                if dice >= 0.5:
                                    y_list[i].append(item)
                if '' in y_list[i]:
                    y_list[i].remove('')
                if y_list[i]:
                    y_list[i].sort(key=lambda x: np.abs(len(x)-len(x_list[i]))/len(x))
                    y_list[i] = [y_list[i][0]]

        elif re.findall('[a-zA-Z- ]{4,}', x_list[i]) and (not y_list[i] or y_list[i] == ['']):
            for item in standard_icd:
                if re.findall('[a-zA-Z- ]{4,}', item):
                    eng = re.findall('[a-zA-Z- ]{4,}', item)
                    eng_x = re.findall('[a-zA-Z- ]{4,}', x_list[i])
                    if eng_x[0].replace('-','').lower() == eng[0].replace('-','').lower():
                        y_list[i].append(item)
            if '' in y_list[i]:
                y_list[i].remove('')
            if y_list[i]:
                y_list[i].sort(key=lambda x: np.abs(len(x)-len(x_list[i]))/len(x))
                y_list[i] = [y_list[i][0]]
        
        # if not y_list[i] or y_list[i] == ['']:
        #     for item in standard_icd:
        #         x = clean(x_list[i])
        #         y = clean(item)
        #     if '' in y_list[i]:
        #         y_list[i].remove('')
        #     if y_list[i]:
        #         y_list[i].sort(key=lambda x: np.abs(len(x)-len(x_list[i]))/len(x))
        #         y_list[i] = [y_list[i][0]]

    return y_list



if __name__ == '__main__':

    with open('./temp_tmp.txt', 'r') as f:
        lines = f.readlines()
    x_list = []
    y_list = []
    for line in lines:
        l = line.strip('\n').split("\t")
        x_list.append(l[0])
        y_list.append(l[1].split("##"))
    new_y_list = copy.deepcopy(y_list)
    clean2standard, origin_y, standard2clean, cleaned_y = load_icd()
    # with open('./icd.pkl', 'wb') as f:
    #     pickle.dump(origin_y, f)
    # with open('./icd.pkl', 'rb') as f:
    #     origin_y = pickle.load(f)

    # x_list = ['RamsayHunt综合征左侧面部']
    # y_list = [[]]
    # origin_y = ['肌阵挛小脑性共济失调[Ramsay-Hunt综合征]']
    new_y_list = copy.deepcopy(y_list)
    out_y = syndrome(x_list, y_list, origin_y)
    with open('../final_standard_out_tmp.txt', 'w') as f:
        for i in range(len(x_list)):
            strrr = ''
            for item in out_y[i]:
                strrr = strrr + item + '##'
            f.write(x_list[i]+'\t'+strrr[:-2]+'\n')
