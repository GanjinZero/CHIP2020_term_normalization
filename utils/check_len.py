import numpy as np
import os
import pandas as pd

with open("../final_standard_out.txt", "r") as f:
    real_lines = f.readlines()

real_x = [line.strip() for line in real_lines]

if os.path.exists("ICD_10v601.csv"):
    df = pd.read_csv("ICD_10v601.csv", header=None)
else:
    df = pd.read_csv("../ICD_10v601.csv", header=None)
origin_y = set(df[1].tolist())

#print(df.loc[0:5, 1])



file_name = "../f_1.txt"
with open(file_name, "r") as f:
    lines = f.readlines()

y = [line.strip('\n').split('\t')[1].split('##') for line in lines]
l = [0 if '' in yy else len(yy) for yy in y]
print(np.bincount(l))

'''
file_name = "../f_3.txt"
with open(file_name, "r") as f:
    lines = f.readlines()

x = [line.strip('\n').split('\t')[0] for line in lines]

for i in range(len(x)):
    if x[i] != real_x[i]:
        print(x[i], real_x[i])


y = [line.strip('\n').split('\t')[1].split('##') for line in lines]
l = [0 if '' in yy else len(yy) for yy in y]
print(np.bincount(l))


for yy in y:
    for yyy in yy :
        if yyy != '' and yyy not in origin_y:
            print(yyy)



file_name = "../f_4.txt"
with open(file_name, "r") as f:
    lines = f.readlines()

y = [line.strip('\n').split('\t')[1].split('##') for line in lines]
l = [0 if '' in yy else len(yy) for yy in y]
print(np.bincount(l))

x = [line.strip('\n').split('\t')[0] for line in lines]

for i in range(len(x)):
    if x[i] != real_x[i]:
        print(x[i], real_x[i])

for yy in y:
    for yyy in yy :
        if yyy != '' and yyy not in origin_y:
            print(yyy)

for yy in origin_y:
    if "\"" in yy:
        print(yy)
        break

'''