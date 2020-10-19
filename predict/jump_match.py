import sys
sys.path.append("../")
from rule_base.split_v2 import split
#from pypinyin import pinyin, Style
from tqdm import tqdm


def jump_find(x, y, jump_max=-1, min_len_y=-1):
    # 冠状动脉粥样症前降支心肌桥	冠状动脉肌桥
    punc_set = set("、，。,.；;()（）【】\"\'")

    y_split = [y]
    y_another = y.find("[")
    if y_another >= 0 and y[-1] == "]":
        y_split = [y[0:y_another], y[y_another + 1:-1]]
    for y in y_split:
        if x == y:
            return True
        if not y or len(y) < min_len_y:
            continue
        if y[-1] == "]":
            y = y[:-1]
        idx_x = 0
        idx_y = 0
        len_x = len(x)
        len_y = len(y)
        last_x = 1000
        use_punc_set = set([punc for punc in punc_set if not punc in y])
        while idx_x < len_x:
            if x[idx_x] in use_punc_set or x[idx_x] == " ":
                idx_x += 1
                idx_y = 0
                continue
            if x[idx_x] == y[idx_y]:
            #if x[idx_x] == y[idx_y] or pinyin_x[idx_x] == pinyin_y[idx_y]:
                if jump_max >= 0:
                    if idx_x - last_x > jump_max:
                        return False
                    last_x = idx_x
                idx_y += 1
            if idx_y == len_y:
                return True
            idx_x += 1
    return False


def jump_match(x_list, cleaned_y, clean2standard, jump_max=-1, min_len_y=-1):
    pred_y = []
    for x in tqdm(x_list):
        now_y = []
        split_x = split(x)
        for xx in split_x:
            for y in cleaned_y:
                if y == xx:
                    now_y.append(clean2standard[y])
                elif len(y) >= min_len_y:
                    if xx.find(y[0]) >= 0:
                        if xx.find(y) >= 0:
                            now_y.append(clean2standard[y])
                        elif jump_find(xx, y, jump_max):
                            now_y.append(clean2standard[y])
        pred_y.append(now_y)
    return pred_y
