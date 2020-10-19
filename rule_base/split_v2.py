punc_set = set("、，。,.；;()（）[]【】\"\'?？ ：")
def split(string, return_set=False):
    # , ，  ; . 。 ? whitespace 、 \ ：
    opt = []
    now = ""
    for idx, ch in enumerate(string):
        if ch in punc_set:
            if now:
                if return_set:
                    opt.append(set(now))
                else:
                    opt.append(now)
                now = ""
        else:
            now = now + ch
    if now:
        if return_set:
            opt.append(set(now))
        else:
            opt.append(now)
    return opt