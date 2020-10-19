import re

def reject_level(x, y):
    pattern_y = re.findall('[1-8][期度型级]', y)
    pattern_x = re.findall('[1-8][期度型级]', x)
    raw_num =  re.findall('[^0-9a-z][1-8][^0-9期度型级周年日点]', x)
    for item in raw_num:
        pattern_x += re.findall('[1-8]', item)
    for item in pattern_y:
        if item in pattern_x or item[0] in pattern_x:
            continue
        else:
            return True
    return False