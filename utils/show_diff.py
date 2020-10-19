import sys

input_file_1 = sys.argv[1]
input_file_2 = sys.argv[2]
output_file = sys.argv[3]

def sort(lines):
    return ["##".join(sorted(line.split("##"))) for line in lines]

with open(input_file_1, "r",  encoding="utf-8") as f:
    input_lines_1 = f.readlines()
    x_list = [line.split("\t")[0] for line in input_lines_1]
    input_lines_1 = [line.split("\t")[1].strip() for line in input_lines_1]
    input_lines_1 = sort(input_lines_1)

with open(input_file_2, "r",  encoding="utf-8") as f:
    input_lines_2 = f.readlines()
    input_lines_2 = [line.split("\t")[1].strip() for line in input_lines_2]
    input_lines_2 = sort(input_lines_2)

with open(output_file, "w",  encoding="utf-8") as f:
    for idx, x in enumerate(x_list):
        if input_lines_1[idx] != input_lines_2[idx]:
            f.write(x + "| " + input_lines_1[idx] + "| " + input_lines_2[idx] + "\n")
