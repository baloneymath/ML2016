#! /usr/bin/env python

import sys

model = sys.argv[1]
test_file = sys.argv[2]


z = 0
def resetz():
    global z
    z = 0

m_table = []
temp = []
with open(model, 'r') as f:
    for line in f:
        tokens = line.split(',')
        if len(tokens) == 1:
            if len(temp) > 0:
                m_table.append(temp)
            temp = []
        else:
            temp.append(tokens)

class Node(object):
    def __init__(self, params, c):
        self.idx = params[0]
        self.l_idx = params[1]
        self.r_idx = params[2]
        self.label = params[3]
        self.feature = params[4]
        self.sp = params[5]
        self.left = None
        self.right = None
        self.c = c
    def creat_child(self, params, flag):
        if flag == 'l':
            self.left = Node(params, self.c)
        elif flag == 'r':
            self.right = Node(params, self.c)

    def rebuild(self):
        global z
        z += 1
        if self.label == '-1':
            self.creat_child(m_table[self.c][z], 'l')
            self.left.rebuild()
            self.creat_child(m_table[self.c][z], 'r')
            self.right.rebuild()
    
    def find(self, sample):
        if self.label == '-1':
            if sample[int(self.feature)] < float(self.sp):
                return self.left.find(sample)
            else:
                return self.right.find(sample)
        else:
            return self

class Tree(Node):
    def __init__(self, params, i):
        self.root = Node(params, i)
        self.root.rebuild()
        resetz()
    def compare(self, sample):
        a = self.root.find(sample)
        return float(a.label)

Trees = []
for i in range(len(m_table)):
    T = Tree(m_table[i][0], i)
    Trees.append(T)

test_table = []
with open(test_file, 'r') as f:
    for line in f:
        tmp = []
        tokens = line.split(',')
        for i in range(len(tokens) - 1):
            tmp.append(float(tokens[i+1]))
        test_table.append(tmp)

out = [-1] * len(test_table)
for i in range(len(test_table)):
    value = 0.
    for j in range(len(Trees)):
        value += Trees[j].compare(test_table[i])
    if value / len(m_table) > 0.5:
        out[i] = 1
    else:
        out[i] = 0

ofile = open(sys.argv[3], 'w')
ofile.write("id,label")
ofile.write('\n')

for i in range(len(test_table)):
    ofile.write(str(i+1) + ',' + str(out[i]))
    ofile.write('\n')
ofile.close()
