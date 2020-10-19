import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open("../data/test.txt", "r", encoding="utf-8") as f:
    x_lines = f.readlines()

with open(input_file, "r",  encoding="utf-8") as f:
    result_lines = f.readlines()

with open(output_file, "w",  encoding="utf-8") as f:
    for idx, line in enumerate(x_lines):
        y = result_lines[idx].strip().split("\t")
        if len(y) == 2:
            y = y[-1]
        else:
            y = ""
        y = "##".join(["\"" + yy + "\"" if yy.find(",")>=0 else yy for yy in y.split("##")])
        f.write(line.strip() + "\t" + y + "\n")