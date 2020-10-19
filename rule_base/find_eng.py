from load_icd_v2 import load_icd, load_train_icd

x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list = load_train_icd()
clean2standard, origin_y, standard2clean = load_icd()

# x_list, clean2standard.keys()
import re

eng_dict = {}
for line in x_list:
    res = re.findall("[a-zA-Z]+", line)
    for w in res:
        if not w in eng_dict:
            eng_dict[w] = 0
        eng_dict[w] += 1

eng_dict={w:eng_dict[w] for w in eng_dict if len(w)>1 and eng_dict[w]==1}
print(eng_dict)
print(len(eng_dict))

"""
{'iiia': 10, 'b': 270, 's': 30, 'st': 17, 'pci': 2, 'rca': 1, 'oa': 38, 'nt': 1,
'tbsa': 5, 'cini': 11, 'iga': 17, 'esd': 3, 'i': 80, 'g': 57, 'p': 26, 'l': 67,
'psa': 5, 'ypt': 3, 'n': 80, 'm': 150, 'cad': 2, 'ca': 49, 'pt': 45, 'an': 13,
'nerd': 1, 'ed': 2, 'iib': 22, 'sonk': 1, 'cabg': 1, 'cashang': 1, 't': 216,
'ii': 56, 'burrkitt': 1, 'iv': 70, 'iiic': 18, 'ct': 23, 'bn': 7, 'ps': 4, 'egfr': 5,
'caa': 2, 'chdhf': 1, 'sle': 2, 'loa': 17, 'ceaca': 1, 'mp': 2, 'a': 110, 'iii': 58,
'cm': 10, 'iva': 5, 'ou': 3, 'kappa': 3, 'd': 52, 'iss': 11, 'ciniii': 6, 'mt': 10,
'iic': 1, 'rr': 1, 'ep': 1, 'cin': 13, 'iihpv': 1, 'cinii': 3, 'ckd': 7, 'paget': 12,
'simmond': 1, 'tpsa': 2, 'v': 10, 'pd': 4, 'ddh': 2, 'c': 103, 'ebstein': 3, 'tc': 2,
'db': 2, 'mgd': 1, 'ib': 10, 'fl': 1, 'teson': 1, 'mm': 3, 'pbc': 2, 'f': 7, 'fgr': 2,
'ra': 1, 'gd': 1, 'ctxnxm': 1, 'bivb': 1, 'r': 4, 'poland': 2, 'poos': 1, 'eb': 7,
'barrett': 5, 'msi': 1, 'nyha': 7, 'mods': 5, 'pnet': 1, 'killip': 9,
'luminalbher': 1, 'lutembacher': 2, 'deg': 1, 'mr': 9, 'tia': 5, 'acl': 1,
'chiari': 4, 'amx': 1, 'acth': 2, 'who': 4, 'he': 2, 'hiv': 41, 'iia': 9, 'bowen': 6,
'rathke': 2, 'igg': 6, 'as': 2, 'hp': 4, 'vkh': 1, 'k': 17, 'dic': 2, 'hcc': 4, 'le': 1,
'nyhaiii': 1, 'kt': 1, 'rhcc': 1, 'castleman': 6, 'ggo': 2, 'lge': 2, 'de': 1,
'bakey': 1, 'poncet': 1, 'lst': 1, 'txnxm': 1, 'cgnfsgs': 1, 'bm': 2, 'gdma': 1,
'rsa': 1, 'ht': 1, 'nk': 10, 'abo': 8, 'htn': 1, 'cushing': 3, 'ds': 7, 'ncr': 1,
'caroli': 4, 'iiib': 14, 'hunt': 3, 'h': 7, 'ddd': 1, 'ln': 2, 'alzheimer': 1, 'prl': 3,
'nsclc': 1, 'evens': 1, 'mikulicz': 1, 'fljc': 1, 'ev': 2, 'igd': 3, 'lambda': 2,
'copd': 4, 'chf': 1, 'killipi': 3, 'od': 1, 'ipi': 1, 'mia': 2, 'dieulafoy': 1,
'x': 22, 'lad': 1, 'ao': 1, 'hoffa': 2, 'cea': 3, 'hpv': 6, 'tunner': 1, 'roa': 5,
'carcinoma': 2, 'po': 1, 'w': 3, 'pagtes': 1, 'nkt': 1, 'killipii': 2, 'addison': 2,
'rett': 1, 'iud': 3, 'wernicke': 2, 'castelman': 1, 'll': 2, 'goldenhar': 2, 'kras': 1,
'am': 2, 'rh': 9, 'ufh': 1, 'ddhcroweiv': 1, 'stevens': 1, 'johnson': 2, 'athmas': 1,
'aecopd': 2, 'mcr': 1, 'rfa': 1, 'lo': 1, 'kippip': 1, 'mo': 3, 'pancreaticcancer': 1,
'gvhd': 3, 'srs': 1, 'gmxs': 1, 'e': 33, 'ama': 1, 'cgd': 1, 'msd': 1, 'bmt': 1, 'cr': 4,
'sd': 2, 'j': 5, 'ic': 1, 'arnold': 2, 'crohn': 3, 'gdm': 1, 'tb': 1, 'dvt': 1,
'dixon': 2, 'sturge': 2, 'weber': 4, 'mx': 2, 'luminal': 1, 'smtca': 1, 'eige': 1,
'mtlater': 1, 'nse': 1, 'psvt': 1, 'nib': 1, 'pmmr': 1, 'z': 2, 'gaucher': 1, 'fai': 1,
'lop': 1, 'lot': 1, 'pcos': 1, 'sweet': 1, 'nos': 4, 'graves': 3, 'ia': 3, 'ad': 2,
'cdh': 1, 'nxmx': 1, 'enneking': 1, 'leep': 1, 'caiia': 1, 'ice': 3, 'itp': 1, 'wbc': 1,
'malt': 1, 'naoh': 1, 'fd': 1, 'ck': 1, 'hl': 1, 'xxrob': 1, 'tip': 1, 'chb': 1,
'est': 1, 'enbd': 1, 'tsh': 3, 'carolis': 1, 'rt': 1, 'cn': 2, 'ivb': 1, 'lam': 1,
'ptca': 3, 'ptcd': 1, 'alk': 4, 'hunter': 1, 'pof': 1, 'ems': 1, 'biv': 1, 'asd': 1,
'vsd': 1, 'pda': 1, 'ivv': 1, 'txn': 1, 'ivf': 3, 'her': 4, 'stills': 1, 'abc': 1,
'ecog': 1, 'flipi': 2, 'castlemen': 1, 'cgvhd': 1, 'ladladpci': 1, 'ards': 1, 'op': 1,
'tbca': 1, 'lsa': 1, 'afp': 1, 'sclc': 1, 'ecg': 1, 'zi': 1, 'pdl': 1, 'mss': 1,
'masson': 1, 'ai': 1, 'ms': 2, 'tctg': 1, 'q': 25, 'cmt': 1, 'rop': 2, 'plus': 2,
'ph': 4, 'dlbcl': 1, 'turner': 2, 'aml': 4, 'pta': 1, 'alpers': 1, 'tat': 1, 'fi': 1,
'cavc': 1, 'troca': 1, 'avnrt': 2, 'cg': 2, 'lmp': 2, 'cavccoa': 1, 'ggt': 1,
'edss': 1, 'avn': 2, 'vin': 3, 'ebv': 3, 'dcis': 1, 'chd': 1, 'gu': 1, 'viniii': 2,
'terson': 2, 'cloves': 1, 'cre': 1, 'hlv': 4, 'marie': 2, 'bamberger': 1, 'crigler': 1,
'najjar': 1, 'brown': 1, 'beta': 31, 'addisons': 1, 'nec': 1, 'nsaid': 2, 'igdlambda': 1,
'byler': 1, 'gastaut': 2, 'brusting': 1, 'perrry': 1, 'nmda': 1, 'crouzon': 1,
'igglambda': 1, 'cd': 9, 'still': 1, 'di': 1, 'george': 1, 'kayser': 1,
'fleischer': 1, 'xy': 5, 'gamma': 7, 'xx': 7, 'igm': 6, 'alpha': 18, 'se': 1,
'brill': 1, 'zinsser': 1, 'delta': 9, 'sotos': 1, 'killipiv': 1, 'lga': 1,
'zollinger': 1, 'ellison': 1, 'brugada': 2, 'friedreich': 3, 'lsd': 2, 'kaschin': 1,
'beck': 1, 'emery': 1, 'dreifuss': 1, 'apud': 1, 'prune': 1, 'belly': 1, 'scid': 2,
'alport': 1, 'bernard': 1, 'soulier': 1, 'lengre': 1, 'bcg': 3, 'alagille': 1,
'mendelson': 1, 'sars': 1, 'mooren': 1, 'wolf': 1, 'hirschorn': 1, 'dandy': 1,
'walker': 1, 'dubowitz': 1, 'ki': 1, 'robinow': 1, 'silverman': 1, 'smith': 2,
'west': 1, 'wegeners': 1, 'granulomatosis': 1, 'cockayne': 1, 'dtp': 3, 'wiskott': 1, 'aldrich': 1, 'qt': 3, 'scheuermann': 1, 'moebius': 1, 'aarskog': 1, 'minkowski': 1, 'chauffard': 1, 'mg': 9, 'ml': 9, 'pyle': 1, 'proteus': 1, 'duchenne': 1, 'bcr': 3, 'abl': 3, 'lfa': 1, 'vonhippel': 1, 'lindau': 1, 'ss': 1, 'richter': 1, 'clippers': 1, 'tab': 3, 'budd': 1, 'mirrizi': 1, 'shwachman': 1, 'diamond': 1, 'ollier': 1, 'esbls': 2, 'buerger': 1, 'vi': 1, 'cyclops': 1, 'liddle': 1, 'peter': 1, 'gilbert': 1, 'jaccoud': 1, 'gm': 5, 'hermansky': 1, 'pudlak': 1, 'lyell': 1, 'cbf': 3, 'eto': 2, 'asperger': 1, 'eo': 1, 'albright': 2, 'nezelof': 1, 'gorlin': 1, 'vainiii': 1, 'pnp': 1, 'ada': 1, 'balo': 1, 'meige': 1, 'hurler': 1, 'rotor': 1, 'ige': 1, 'louis': 1, 'bar': 1, 'duane': 1, 'seckel': 1, 'bb': 2, 'xq': 2, 'bl': 1, 'vogt': 1, 'fazio': 1, 'londe': 1, 'cpk': 1, 'ab': 2, 'whipple': 1, 'russell': 1, 'silver': 1, 'wilson': 1, 'laennec': 1, 'behcet': 1, 'leopard': 1, 'marchesani': 1, 'igalambda': 1, 'ehlers': 1, 'danlos': 1, 'larsen': 1, 'pickwickian': 1, 'stein': 1, 'leventhal': 1, 'xxx': 2, 'eales': 1, 'pvc': 1, 'adie': 1, 'lowe': 1, 'treachercollins': 1, 'coo': 1, 'tar': 1, 'potter': 1, 'blau': 1, 'tinu': 1, 'patau': 1, 'marinesco': 1, 'sjogren': 1, 'christian': 1, 'inv': 1, 'rubinstein': 1, 'taybi': 1, 'coats': 1, 'hellp': 1, 'leber': 1, 'kartagener': 1, 'pml': 2, 'rar': 2, 'legg': 1, 'calve': 1, 'perthes': 1, 'aild': 1, 'panayiotopoulos': 1, 'strumpell': 1, 'lorrain': 1, 'gammadeltat': 2, 'becker': 1, 'dressler': 1, 'nelaton': 1, 'lesch': 1, 'nyhan': 1, 'tay': 1, 'sachs': 1, 'prader': 1, 'willi': 1, 'pilon': 2, 'apert': 1, 'bedkwith': 1, 'wiedemann': 1, 'dna': 2, 'dubin': 1, 'vii': 1, 'vonrecklinghausen': 1, 'hurst': 1, 'slap': 1, 'tietze': 1, 'hsv': 2, 'dish': 1, 'stokes': 1, 'adams': 1, 'kupffer': 1, 'denys': 1, 'drash': 1, 'kabuki': 1, 'klippel': 1, 'trenaunay': 1, 'tt': 1, 'gradenigo': 1, 'kearn': 1, 'sayre': 1, 'dravet': 1, 'anca': 2, 'hps': 1, 'hcps': 1, 'tipss': 1, 'lemli': 1, 'opitz': 1, 'retts': 1, 'takayasu': 1, 'hangman': 1, 'dequervain': 1, 'mll': 1, 'marcusgunn': 1, 'jk': 1, 'rieger': 1, 'crest': 1, 'sc': 1, 'kimurus': 1, 'terrien': 1, 'leriche': 1, 'sandhoff': 1, 'conn': 1, 'felty': 1, 'schilder': 1, 'pierre': 1, 'robin': 1, 'bt': 1, 'didmoad': 1, 'morvan': 1, 'mrsa': 1, 'killipiii': 1, 'goodpasture': 1, 'weaver': 1, 'crpa': 1, 'kawasaki': 1, 'rokitansky': 1, 'kustner': 1, 'hauser': 1, 'waardenburg': 1, 'kelly': 1, 'parerson': 1, 'kikuchi': 1, 'meckel': 1, 'gruber': 1, 'nsaids': 1, 'scld': 1, 'chediak': 1, 'steinbrinck': 1, 'higashi': 1, 'miller': 1, 'fisher': 1, 'rosai': 1, 'dorfman': 1, 'xyy': 2, 'klinefelter': 1, 'guerin': 1, 'stern': 1, 'fahr': 1, 'lightwood': 1, 'hallerman': 1, 'streiff': 1, 'ellis': 1, 'vancreveld': 1, 'irvan': 1, 'marfan': 1, 'frasier': 1, 'stargardt': 1, 'jacksong': 1, 'cantrell': 1, 'cockett': 1, 'lennox': 1, 'forestier': 1, 'cross': 1, 'acs': 1, 'thomsen': 1, 'ac': 1, 'mds': 1, 'vater': 1, 'bell': 1, 'laurence': 1, 'moon': 1, 'biedl': 1, 'reye': 1, 'angelman': 1, 'noonan': 1, 'velo': 1, 'cardio': 1, 'facia': 1, 'corneliadelange': 1, 'crab': 1, 'danon': 1, 'gitelman': 1, 'down': 1, 'blackfan': 1, 'diamong': 1, 'williams': 1, 'qrs': 2, 'sapho': 1, 'ramsy': 1, 'nadh': 1, 'waterhouse': 1, 'friderichsen': 1, 'libman': 1, 'sacks': 1, 'plummer': 1, 'vinson': 1, 'churg': 1, 'strauss': 1, 'lgg': 1, 'morton': 1, 'efa': 1, 'rasmussen': 1, 'myh': 1, 'vre': 1, 'ldh': 1, 'bartter': 1, 'lisfranc': 1, 'ganser': 1, 'hpfh': 1, 'fuchs': 3, 'menkes': 1, 'mu': 2, 'landau': 1, 'kleffner': 1, 'ramsay': 1, 'medpor': 1, 'kleine': 1, 'levin': 1, 'vanderwoude': 1, 'dvd': 1, 'lx': 1, 'refsum': 1, 'alstrom': 1, 'bloom': 1, 'peutz': 1, 'jeghers': 1, 'imerslund': 1, 'helveston': 1, 'todd': 1, 'ptld': 1, 'edwards': 1, 'retraction': 1, 'of': 1, 'mediastinum': 1, 'fanconi': 1, 'xxy': 1, 'omenn': 1, 'lemmel': 1, 'brock': 1, 'lee': 1, 'zellweger': 1, 'reiter': 1, 'mmr': 1, 'htlv': 1, 'maffucci': 1}
"""

"""
{'iiia': 10, 'st': 12, 'pci': 2, 'oa': 38, 'tbsa': 5, 'cini': 11, 'iga': 7, 'esd': 3,
'psa': 4, 'ypt': 3, 'cad': 2, 'ca': 47, 'pt': 45, 'an': 13, 'ed': 2, 'iib': 21, 'ii': 33,
'iv': 59, 'iiic': 18, 'ct': 22, 'bn': 7, 'ps': 4, 'egfr': 5, 'caa': 2, 'sle': 2,
'loa': 17, 'mp': 2, 'iii': 35, 'cm': 10, 'iva': 5, 'ou': 3, 'kappa': 2, 'iss': 5,
'ciniii': 5, 'mt': 10, 'cin': 11, 'cinii': 3, 'ckd': 7, 'paget': 12, 'tpsa': 2,
'pd': 3, 'ddh': 2, 'ebstein': 2, 'tc': 2, 'db': 2, 'ib': 10, 'mm': 3, 'pbc': 2,
'fgr': 2, 'barrett': 5, 'nyha': 2, 'mods': 5, 'killip': 9, 'mr': 9, 'tia': 5,
'chiari': 2, 'acth': 2, 'who': 4, 'hiv': 3, 'iia': 8, 'bowen': 5, 'rathke': 2,
'igg': 2, 'as': 2, 'hp': 4, 'dic': 2, 'hcc': 4, 'castleman': 5, 'ggo': 2, 'bm': 2,
'nk': 3, 'abo': 2, 'cushing': 3, 'caroli': 3, 'iiib': 14, 'ln': 2, 'prl': 3, 'copd': 4,
'killipi': 2, 'mia': 2, 'cea': 2, 'hpv': 5, 'roa': 5, 'carcinoma': 2, 'iud': 3,
'am': 2, 'rh': 2, 'aecopd': 2, 'mo': 3, 'gvhd': 3, 'cr': 3, 'crohn': 3, 'dixon': 2,
'mx': 2, 'nos': 2, 'ia': 3, 'ad': 2, 'ice': 2, 'tsh': 2, 'cn': 2, 'ptca': 3, 'ivf': 3,
'her': 4, 'flipi': 2, 'rop': 2, 'plus': 2, 'avnrt': 2, 'cg': 2, 'lmp': 2, 'avn': 2}
"""

"""
{'rca': 1, 'nt': 1, 'nerd': 1, 'sonk': 1, 'cabg': 1, 'cashang': 1, 'burrkitt': 1,
'chdhf': 1, 'ceaca': 1, 'iic': 1, 'rr': 1, 'ep': 1, 'iihpv': 1, 'simmond': 1, 'mgd': 1,
'fl': 1, 'teson': 1, 'ra': 1, 'gd': 1, 'ctxnxm': 1, 'bivb': 1, 'poland': 1, 'poos': 1,
'eb': 1, 'msi': 1, 'pnet': 1, 'luminalbher': 1, 'lutembacher': 1, 'deg': 1, 'acl': 1,
'amx': 1, 'he': 1, 'vkh': 1, 'le': 1, 'nyhaiii': 1, 'kt': 1, 'rhcc': 1, 'lge': 1,
'de': 1, 'bakey': 1, 'poncet': 1, 'lst': 1, 'txnxm': 1, 'cgnfsgs': 1, 'gdma': 1,
'rsa': 1, 'ht': 1, 'htn': 1, 'ds': 1, 'ncr': 1, 'hunt': 1, 'ddd': 1, 'alzheimer': 1,
'nsclc': 1, 'evens': 1, 'mikulicz': 1, 'fljc': 1, 'ev': 1, 'igd': 1, 'lambda': 1,
'chf': 1, 'od': 1, 'ipi': 1, 'dieulafoy': 1, 'lad': 1, 'ao': 1, 'hoffa': 1, 'tunner': 1,
'po': 1, 'pagtes': 1, 'nkt': 1, 'killipii': 1, 'addison': 1, 'rett': 1, 'wernicke': 1,
'castelman': 1, 'll': 1, 'goldenhar': 1, 'kras': 1, 'ufh': 1, 'ddhcroweiv': 1,
'stevens': 1, 'johnson': 1, 'athmas': 1, 'mcr': 1, 'rfa': 1, 'lo': 1, 'kippip': 1,
'pancreaticcancer': 1, 'srs': 1, 'gmxs': 1, 'ama': 1, 'cgd': 1, 'msd': 1, 'bmt': 1,
'sd': 1, 'ic': 1, 'arnold': 1, 'gdm': 1, 'tb': 1, 'dvt': 1, 'sturge': 1, 'weber': 1,
'luminal': 1, 'smtca': 1, 'eige': 1, 'mtlater': 1, 'nse': 1, 'psvt': 1, 'nib': 1,
'pmmr': 1, 'gaucher': 1, 'fai': 1, 'lop': 1, 'lot': 1, 'pcos': 1, 'sweet': 1,
'graves': 1, 'cdh': 1, 'nxmx': 1, 'enneking': 1, 'leep': 1, 'caiia': 1, 'itp': 1,
'wbc': 1, 'malt': 1, 'naoh': 1, 'fd': 1, 'ck': 1, 'hl': 1, 'xxrob': 1, 'tip': 1,
'chb': 1, 'est': 1, 'enbd': 1, 'carolis': 1, 'rt': 1, 'ivb': 1, 'lam': 1, 'ptcd': 1,
'alk': 1, 'hunter': 1, 'pof': 1, 'ems': 1, 'biv': 1, 'asd': 1, 'vsd': 1, 'pda': 1,
'ivv': 1, 'txn': 1, 'stills': 1, 'abc': 1, 'ecog': 1, 'castlemen': 1, 'cgvhd': 1,
'ladladpci': 1, 'ards': 1, 'op': 1, 'tbca': 1, 'lsa': 1, 'afp': 1, 'sclc': 1, 'ecg': 1,
'zi': 1, 'pdl': 1, 'mss': 1, 'masson': 1, 'ai': 1, 'ms': 1, 'tctg': 1, 'cmt': 1, 'ph': 1,
'dlbcl': 1, 'turner': 1, 'aml': 1, 'pta': 1, 'alpers': 1, 'tat': 1, 'fi': 1, 'cavc': 1,
'troca': 1, 'cavccoa': 1, 'ggt': 1, 'edss': 1, 'vin': 1, 'ebv': 1, 'dcis': 1, 'chd': 1,
'gu': 1, 'viniii': 1, 'terson': 1}
"""
