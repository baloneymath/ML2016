#!/usr/bin/env python3
import sys
f = open("linear_regression.csv", 'r')
f2 = open("ans.csv", 'r')
counter = 0

ff = []
ff2 = []
y = 0.0
for line in f:
    tokens = line.split(',')
    ff.append(tokens[1])
for line in f2:
    tokens = line.split(',')
    ff2.append(tokens[1])
for i in range(1,len(ff)):
    y +=(float(ff[i])-float(ff2[i])) ** 2
y = y/240
y = y ** 0.5
print(y)

