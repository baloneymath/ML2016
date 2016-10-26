#! /usr/bin/env python3

import sys
import math

train_file = sys.argv[1]
my_model = open(sys.argv[2], 'w')

#######################################
train_table = []
with open(train_file, 'r') as f:
    for line in f:
        tokens = line.split(',')
        train_table.append(tokens)
########################################
iteration = 1800
master_stepsize = 1e-1
autocorr = 0.00
epsilon = 0

flag = [True for m in range(57)]

b = 0
grad_B = 0
w = [0] * 57 # 57 features
grad = [0] * 57



def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

def cross_entropy(f, y):
    return -(y * math.log(f + 1e-60) + (1 - y) * math.log(1 - f + 1e-60))

count = 0
for i in range(iteration):
    L = 0
    for j in range(len(train_table)):
        GRAD_B = 0
        GRAD = [0] * 57
        data = [0] * 57
        for k in range(57):
            data[k] = float(train_table[j][k+1])
        y = b # bias
        _y = float(train_table[j][58]) # real data

        for k in range(57):
            if flag[k] == True:
                y += w[k] * data[k]
        f = sigmoid(y)
        L += cross_entropy(f, _y)
        GRAD_B += -(_y - f)
        for k in range(57):
            if flag[k] == True:
                GRAD[k] += -(_y - f) * data[k]

        # gradient descent
        grad_B += GRAD_B ** 2
        if grad_B == 0:
            ada_grad_B = 0
        else:
            ada_grad_B = GRAD_B / (epsilon + grad_B ** 0.5)
        b = b - master_stepsize * ada_grad_B

        ada_grad = [0] * 57
        for k in range(57):
            if flag[k] != True:
                continue
            grad[k] += GRAD[k] ** 2
            if grad[k] == 0:
                ada_grad[k] = 0
            else:
                ada_grad[k] = GRAD[k] / (epsilon + grad[k] ** 0.5)
            w[k] = w[k] - master_stepsize * ada_grad[k]
    count += 1
    if count % 10 == 0:
        acc = 0.0
        for j in range(len(train_table)):
            _y = float(train_table[j][58])
            y = b
            for k in range(57):
                y += w[k] * float(train_table[j][k+1])
            if y > 0.5:
                y = 1
            else:
                y = 0
            if y == _y:
                acc += 1
        acc = acc/4001
        print(count, L, acc)
#######################################################
my_model.write(str(b))
my_model.write('\n')
for i in range(len(w)):
    if i < len(w) - 1:
        my_model.write(str(w[i]))
        my_model.write('\n')
    else:
        my_model.write(str(w[i]))
