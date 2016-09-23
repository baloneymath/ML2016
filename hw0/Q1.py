#!/usr/bin/env python3

import sys

colNum = int(sys.argv[1])
fileName = sys.argv[2]
out = []
f = open(fileName, 'r')
o = open("ans1.txt", 'w')

for line in f:
    temp = filter(lambda x: len(x) > 0, line.split(' '))
    value = list(map(float, temp))
    out.append(value[colNum])

out.sort()

for i in out:
    if i != out[len(out)-1]:
        o.write(str(i) + ',')
o.write(str(out[len(out)-1]))

