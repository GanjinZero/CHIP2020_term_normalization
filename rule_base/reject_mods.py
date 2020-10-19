def reject_mods(x,y_list):
    y_out = []
    # reject
    for item in y_list:
        if "多脏器" in item:
            if "多脏器" in x or "mods" in x or "多器官" in x:
                y_out.append(item)
            else:
                continue
        else:
            y_out.append(item)
    # accept
    if "多脏器功能" in x:
        neg = True
        for item in y_out:
            if "多脏器功能" in item:
                neg = False
                break
        if neg:
            y_out.append("多脏器功能障碍综合征".lower())

    return y_out

if __name__ == "__main__":
    x = "多脏器功能障碍心肝凝血肺胃肠道"
    y_list = ["胃肠功能紊乱"]
    print("Test 1:")
    print(x,'\t','##'.join(y_list))
    print(x,'\t','##'.join(reject_mods(x,y_list)))

    x = "多脏器功能不全循环呼吸胃肠肾脏凝血"
    y_list = ["多脏器功能障碍综合征"]
    print("Test 2:")
    print(x,'\t','##'.join(y_list))
    print(x,'\t','##'.join(reject_mods(x,y_list)))
    
    x = "自身免疫性疾病多脏器受累"
    y_list = ["自身免疫病"]
    print("Test 3:")
    print(x,'\t','##'.join(y_list))
    print(x,'\t','##'.join(reject_mods(x,y_list)))

    x = "急性肝肾凝血等功能衰竭"
    y_list = ["多脏器功能衰竭"]
    print("Test 4:")
    print(x,'\t','##'.join(y_list))
    print(x,'\t','##'.join(reject_mods(x,y_list)))

    x = "多器官功能损伤"
    y_list = ["多脏器功能障碍综合征"]
    print("Test 4:")
    print(x,'\t','##'.join(y_list))
    print(x,'\t','##'.join(reject_mods(x,y_list)))

    x = "mods肝肾心功能损害dic"
    y_list = ["多脏器功能障碍综合征","弥散性血管内凝血"]
    print("Test 4:")
    print(x,'\t','##'.join(y_list))
    print(x,'\t','##'.join(reject_mods(x,y_list)))
