#! /usr/bin/env python3
import sys

model_name = open(sys.argv[1], 'r')
test_file = open(sys.argv[2], 'r')
ofile = open(sys.argv[3], 'w')

w = [0] * 58

i = 0
for line in model_name:
    w[i] = float(line)
    i += 1
model_name.close()

test_table = []
for line in test_file:
    tokens = line.split(',')
    test_table.append(tokens)
test_file.close()

ofile.write("id,label")
ofile.write('\n')

for i in range(len(test_table)):
    y = w[0]
    for j in range(57):
        y += w[j+1] * float(test_table[i][j+1])
    out = 0
    if y >= 0.5:
        out = 1
    else:
        out = 0
    ofile.write(test_table[i][0] + ',')
    ofile.write(str(out))
    ofile.write('\n')
ofile.close()


