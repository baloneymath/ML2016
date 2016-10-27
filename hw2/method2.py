#! /usr/bin/env python

import sys
import math
import random as rd

sys.setrecursionlimit(2048)

train_file = sys.argv[1]

########## setting ###########################
Tree_Num = 128
split_Num = 120
data_range = [2500, 3000]
f_range = [11, 13]
validation = False
n_fold = 5

########## global var ########################
gb = 0
def resetgb():
    global gb
    gb = 0

########## define class ######################
class Node(object):
    def __init__(self, data):
        self.data = data
        self.feature = None
        self.splitpoint = None
        self.left = None
        self.right = None
        self.label = None
        self.index = 0

    def create_child(self, data, flag):
        if flag == 'l':
            self.left = Node(data)
        elif flag == 'r':
            self.right = Node(data)

    def setfeature(self, feature):
        self.feature = feature

    def setsplitpoint(self, splitpoint):
        self.splitpoint = splitpoint

    def setlabel(self, label):
        self.label = label

    def gini(self):
        f_num = rd.randint(f_range[0], f_range[1])
        f = rd.sample(range(0, 57), f_num)
        data = self.data
        index = [0, 0]
        max_con = 1
        for i in f:
            for j in range(len(f_finite[i])):
                lchild = []
                rchild = []
                l_ones = 0.
                r_ones = 0.
                for k in range(len(data)):
                    if data[k][i] < f_finite[i][j]:
                        lchild.append(data[k])
                        if data[k][57] == 1:
                            l_ones += 1
                    else:
                        rchild.append(data[k])
                        if data[k][57] == 1:
                            r_ones += 1
                if len(lchild) == 0 or len(rchild) == 0:
                    continue

                l_t = float(len(lchild))
                r_t = float(len(rchild))
                l_zeros = float(l_t - l_ones)
                r_zeros = float(r_t - r_ones)
                GL = 1 - (l_ones / l_t)**2 - (l_zeros / l_t)**2
                GR = 1 - (r_ones / r_t)**2 - (r_zeros / r_t)**2
                contribute = GL * l_t / len(data) + GR * r_t / len(data)
                if contribute < max_con:
                    index = [i, f_finite[i][j]]
                    max_con = contribute
        if max_con == 1:
            index = [-1, -1]
        self.setfeature(index[0])
        self.setsplitpoint(index[1])
        return index

    def __split(self):
        index = self.gini()
        data = self.data
        if index == [-1, -1]:
            a = 0.
            out = -1
            for i in range(len(data)):
                a += data[i][57]
            a /= len(data)
            if a > 0.5:
                out = 1
            else:
                out = 0
            self.setlabel(out)
            return
        else:
            lchild = []
            rchild = []
            for i in range(len(data)):
                if data[i][index[0]] < index[1]:
                    lchild.append(data[i])
                else:
                    rchild.append(data[i])
            self.create_child(lchild, 'l')
            self.create_child(rchild, 'r')
            self.left.__split()
            self.right.__split()

    def split(self):
        if self.label == None:
            self.__split()

    def find(self, sample):
        if self.label == None:
            if sample[self.feature] < self.splitpoint:
                return self.left.find(sample)
            else:
                return self.right.find(sample)
        else:
            return self

    def setindex(self):
        global gb
        self.index = gb
        gb += 1
        if self.label == None:
            self.left.setindex()
            self.right.setindex()
    def printindex(self):
        print self.index
        if self.label == None:
            self.left.printindex()
            self.right.printindex()

    def recursive_outfile(self, ofile):
        ofile.write(str(self.index) + ',')
        if self.label == None:
            ofile.write(str(self.left.index) + ',')
            ofile.write(str(self.right.index) + ',')
            ofile.write(str(-1) + ',') # self.label
        else:
            ofile.write(str(-99) + ',')
            ofile.write(str(-99) + ',')
            ofile.write(str(self.label) + ',')
        ofile.write(str(self.feature) + ',')
        ofile.write(str(self.splitpoint))
        ofile.write('\n')
        if self.label == None:
            self.left.recursive_outfile(ofile)
            self.right.recursive_outfile(ofile)

class Tree(Node):
    def __init__(self, d_set):
        self.d_set = d_set
        self.root = Node(d_set)
        self.root.split()
        self.root.setindex()
        #self.root.printindex()
        resetgb()
    def compare(self, sample):
        a = self.root.find(sample)
        return a.label
    def recursive_out(self, ofile):
        ofile.write('-')
        ofile.write('\n')
        self.root.recursive_outfile(ofile)
        ofile.write('-')
        ofile.write('\n')

############### parse training data ####################
train_table = []
with open(train_file, 'r') as f:
    for line in f:
        temp = []
        tokens = line.split(',')
        for i in range(len(tokens)-1):
            temp.append(float(tokens[i+1]))
        train_table.append(temp)

############### build split points  ####################
col_ave = [0] * 57
for i in range(57):
    for j in range(len(train_table)):
        col_ave[i] += train_table[j][i]
    col_ave[i] /= len(train_table)
col_sigma = [0] * 57
for i in range(57):
    for j in range(len(train_table)):
        col_sigma[i] += (train_table[j][i] - col_ave[i])**2
    col_sigma[i] /= len(train_table)
    col_sigma[i] = col_sigma[i]**0.5

f_finite = []
for i in range(57):
    temp = [0] * split_Num
    for j in range(split_Num):
        temp[j] = rd.gauss(col_ave[i], col_sigma[i])
    f_finite.append(temp)

############### Grow Trees #############################
Trees = []
for i in range(Tree_Num):
    d_num = rd.randint(data_range[0], data_range[1])
    d_set = rd.sample(range(0, 4001), d_num)
    data = []
    for j in d_set:
        data.append(train_table[j])
    print "Growing Tree:", i+1
    # n-fold validation
    if validation == True:
        T = None
        max_test = 0
        for j in range(n_fold):
            restset = []
            validset = []
            v_num = int(d_num) / 10
            v_set = rd.sample(d_set, v_num)
            for k in d_set:
                if v_set.count(k) == 0:
                    restset.append(train_table[k])
                else:
                    validset.append(train_table[k])
            temp = Tree(restset)
            acc = 0.
            for k in validset:
                value = temp.compare(k)
                if value == k[57]:
                    acc += 1
            acc /= len(validset)
            if acc > max_test:
                T = temp
                max_test = acc
            print acc
        print max_test
        Trees.append(T)
    else:
        Trees.append(Tree(data))

############## Calculate Training score ###############
cnt = 0.
for i in range(len(train_table)):
    value = 0.
    out = -1
    for j in range(len(Trees)):
        value += Trees[j].compare(train_table[i])
    if value / Tree_Num > 0.5:
        out = 1
    else:
        out = 0
    if out == train_table[i][57]:
        cnt += 1
print "Training Acc: ", cnt / len(train_table)


############## Output File ############################
ofile = open(sys.argv[2], 'w')
for i in range(len(Trees)):
    Trees[i].recursive_out(ofile)
print "model builded !!"
