import re


def match_yun(string):
    find = 0
    try:
        y_week = re.search("孕.*?周", string).group()[1:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week))
    except BaseException:
        pass
    try:
        y_week = re.search("孕.*?月", string).group()[1:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week) * 4)
    except BaseException:
        pass
    try:
        y_week = re.search("停经.*?周", string).group()[2:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week))
    except BaseException:
        pass
    try:
        y_week = re.search("停经.*?月", string).group()[2:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week) * 4)
    except BaseException:
        pass
    try:
        y_week = re.search("妊娠.*?周", string).group()[2:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week))
    except BaseException:
        pass
    try:
        y_week = re.search("妊娠.*?月", string).group()[2:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week) * 4)
    except BaseException:
        pass
    try:
        y_week = re.search("妊.*?周", string).group()[1:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week))
    except BaseException:
        pass
    try:
        y_week = re.search("妊.*?月", string).group()[1:-1]
        plus = y_week.find("+")
        if plus >= 0:
            y_week = y_week[0:plus]
        find = max(find, int(y_week) * 4)
    except BaseException:
        pass
    
    tmp = []
    if string.find("双胎"):
        tmp += ["双胎"]

    if 4 >= find >= 1:
        return tmp + ["孕<5周"]
    if 42 >= find >= 5:
        return tmp + ["孕" + str(find) + "周"]
    if 50 >= find >= 43:
        return tmp + ["孕>42周"]
    if find > 50:
        return tmp + ["妊娠状态"]
    return tmp
'''
def test_match_yun():
    """
    p = 0.848
    r = 0.609
    """
    try:
        from load_icd_v2 import load_train_icd
    except BaseException:
        from rule_base.load_icd_v2 import load_train_icd
    x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list = load_train_icd()
    y_yun_pred = []
    y_yun_true = []

    for idx in range(len(x_list)):
        y_yun_true_tmp = []
        if "妊娠状态" in y_icd10_list[idx]:
            y_yun_true_tmp.append("妊娠状态")
        for y in y_icd10_list[idx]:
            yun_find = re.search("孕.*?周", y)
            if yun_find:
                y_yun_true_tmp.append(y)
        find = match_yun(x_list[idx])
        y_yun_pred.append(find)
        y_yun_true.append(y_yun_true_tmp)
        if set(find) != set(y_yun_true_tmp):
            print(y_yun_true_tmp, x_list[idx], find)

    acc(y_yun_pred, y_yun_true)

    return None
'''
def match(string):
    res = match_yun(string)
    return res

if __name__ == "__main__":
    #test_match_yun()
    print(match_yun('妊娠41+3周，孕2产1，单活胎，头位'))