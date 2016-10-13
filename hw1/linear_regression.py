#!/usr/bin/env python3

import sys
import codecs # for python3 utf-8 decode
import math
#import numpy as np
from decimal import *

train = sys.argv[1]
test = open(sys.argv[2], 'r')
ofile = open("linear_regression.csv", 'w')

table = []
a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
with codecs.open(train,"r",encoding='utf-8',errors='ignore') as f:
    for line in f:
        tokens = line.split(',')
        if tokens[2] == 'AMB_TEMP':
            for i in range(3, len(tokens)):
                a0.append(tokens[i])
        elif tokens[2] == 'CH4':
            for i in range(3, len(tokens)):
                a1.append(tokens[i])
        elif tokens[2] == 'CO':
            for i in range(3, len(tokens)):
                a2.append(tokens[i])
        elif tokens[2] == 'NMHC':
            for i in range(3, len(tokens)):
                a3.append(tokens[i])
        elif tokens[2] == 'NO':
            for i in range(3, len(tokens)):
                a4.append(tokens[i])
        elif tokens[2] == 'NO2':
            for i in range(3, len(tokens)):
                a5.append(tokens[i])
        elif tokens[2] == 'NOx':
            for i in range(3, len(tokens)):
                a6.append(tokens[i])
        elif tokens[2] == 'O3':
            for i in range(3, len(tokens)):
                a7.append(tokens[i])
        elif tokens[2] == 'PM10':
            for i in range(3, len(tokens)):
                a8.append(tokens[i])
        elif tokens[2] == 'PM2.5':
            for i in range(3, len(tokens)):
                a9.append(tokens[i])
        elif tokens[2] == 'RH':
            for i in range(3, len(tokens)):
                a11.append(tokens[i])
        elif tokens[2] == 'SO2':
            for i in range(3, len(tokens)):
                a12.append(tokens[i])
        elif tokens[2] == 'THC':
            for i in range(3, len(tokens)):
                a13.append(tokens[i])
        elif tokens[2] == 'WD_HR':
            for i in range(3, len(tokens)):
                a14.append(tokens[i])
        elif tokens[2] == 'WIND_DIREC':
            for i in range(3, len(tokens)):
                a15.append(tokens[i])
        elif tokens[2] == 'WIND_SPEED':
            for i in range(3, len(tokens)):
                a16.append(tokens[i])
        elif tokens[2] == 'WS_HR':
            for i in range(3, len(tokens)):
                a17.append(tokens[i])
        elif (tokens[2] == 'RAINFALL'):
            for i in range(len(tokens)):
                if tokens[i] == 'NR':
                    tokens[i] = 0
                elif tokens[i] == 'NR\r\n':
                    tokens[i] = 0
            for i in range(3, len(tokens)):
                a10.append(tokens[i])
table.append(a0)
table.append(a1)
table.append(a2)
table.append(a3)
table.append(a4)
table.append(a5)
table.append(a6)
table.append(a7)
table.append(a8)
table.append(a9)
table.append(a10)
table.append(a11)
table.append(a12)
table.append(a13)
table.append(a14)
table.append(a15)
table.append(a16)
table.append(a17)
####################################################

iteration = 10000
master_stepsize = 10
fudge_factor = 0
autocorr = 0
alpha = 0.000
stop_point = 1e-2
ol ,nl = 0, 0

b = 0
historical_grad_B = 0

back_2 = 4
back_10_2 = 2

back =            9
back_10 =         5
back_RH =         0
back_NO =         0
back_NO2 =        0
back_NOx =        0
back_O3 =         3
back_SO2 =        0
back_CO =         0
back_THC =        0
back_NMHC =       0
back_WD_HR =      0
back_WIND_DIREC = 0
back_WIND_SPEED = 0
back_WS_HR =      0
back_AMB_TEMP =   0
back_CH4 =        0
back_RAINFALL =   2

back_max = max(back,back_10,back_RH,back_NO,back_NO2,back_NOx,back_O3,
        back_SO2,back_CO,back_THC,back_NMHC,back_WD_HR,back_WIND_DIREC,back_WIND_SPEED,
        back_WS_HR,back_AMB_TEMP,back_CH4,back_RAINFALL)

# PM2.5
data = [0] * back
term = [0] * back
grad = [0] * back # grad
test_data = [0] * back
data_2 = [0] * back_2
term_2 = [0] * back_2
grad_2 = [0] * back_2
test_data_2 = [0] * back_2

# PM10
data_10 = [0] * back_10
term_10 = [0] * back_10
grad_10 = [0] * back_10
test_data_10 = [0] * back_10
data_10_2 = [0] * back_10_2
term_10_2 = [0] * back_10_2
grad_10_2 = [0] * back_10_2
test_data_10_2 = [0] * back_10_2

# RH
data_RH = [0] * back_RH
term_RH = [0] * back_RH
grad_RH = [0] * back_RH
test_data_RH = [0] * back_RH

# NO
data_NO = [0] * back_NO
term_NO = [0] * back_NO
grad_NO = [0] * back_NO
test_data_NO = [0] * back_NO

# NO2
data_NO2 = [0] * back_NO2
term_NO2 = [0] * back_NO2
grad_NO2 = [0] * back_NO2
test_data_NO2 = [0] * back_NO2

# NOx
data_NOx = [0] * back_NOx
term_NOx = [0] * back_NOx
grad_NOx = [0] * back_NOx
test_data_NOx = [0] * back_NOx

# O3
data_O3 = [0] * back_O3
term_O3 = [0] * back_O3
grad_O3 = [0] * back_O3
test_data_O3 = [0] * back_O3

# SO2
data_SO2 = [0] * back_SO2
term_SO2 = [0] * back_SO2
grad_SO2 = [0] * back_SO2
test_data_SO2 = [0] * back_SO2

# CO
data_CO = [0] * back_CO
term_CO = [0] * back_CO
grad_CO = [0] * back_CO
test_data_CO = [0] * back_CO

# THC
data_THC = [0] * back_THC
term_THC = [0] * back_THC
grad_THC = [0] * back_THC
test_data_THC = [0] * back_THC

# NHMC
data_NMHC = [0] * back_NMHC
term_NMHC = [0] * back_NMHC
grad_NMHC = [0] * back_NMHC
test_data_NMHC = [0] * back_NMHC

# WD_HR
data_WD_HR = [0] * back_WD_HR
term_WD_HR = [0] * back_WD_HR
grad_WD_HR = [0] * back_WD_HR
test_data_WD_HR = [0] * back_WD_HR

# WIND_DIREC
data_WIND_DIREC = [0] * back_WIND_DIREC
term_WIND_DIREC = [0] * back_WIND_DIREC
grad_WIND_DIREC = [0] * back_WIND_DIREC
test_data_WIND_DIREC = [0] * back_WIND_DIREC


# WIND_SPEED
data_WIND_SPEED = [0] * back_WIND_SPEED
term_WIND_SPEED = [0] * back_WIND_SPEED
grad_WIND_SPEED = [0] * back_WIND_SPEED
test_data_WIND_SPEED = [0] * back_WIND_SPEED

# WS_HR
data_WS_HR = [0] * back_WS_HR
term_WS_HR = [0] * back_WS_HR
grad_WS_HR = [0] * back_WS_HR
test_data_WS_HR = [0] * back_WS_HR

# AMB_TEMP
data_AMB_TEMP = [0] * back_AMB_TEMP
term_AMB_TEMP = [0] * back_AMB_TEMP
grad_AMB_TEMP = [0] * back_AMB_TEMP
test_data_AMB_TEMP = [0] * back_AMB_TEMP

# CH4
data_CH4 = [0] * back_CH4
term_CH4 = [0] * back_CH4
grad_CH4 = [0] * back_CH4
test_data_CH4 = [0] * back_CH4

# RAINFALL
data_RAINFALL = [0] * back_RAINFALL
term_RAINFALL = [0] * back_RAINFALL
grad_RAINFALL = [0] * back_RAINFALL
test_data_RAINFALL = [0] * back_RAINFALL

z = 0
h= 0
while True:
#for t in range(iteration):
    count = 0
    B = 0
    L = 0
    z += 1

    _y = 0
    L = 0
    TERM = [0] * back # store grad
    TERM_2 = [0] * back_2
    TERM_10 = [0] * back_10 # store grad
    TERM_10_2 = [0] * back_10_2
    TERM_RH = [0] * back_RH
    TERM_NO = [0] * back_NO
    TERM_NO2 = [0] * back_NO2
    TERM_NOx = [0] * back_NOx
    TERM_O3 = [0] * back_O3
    TERM_SO2 = [0] * back_SO2
    TERM_CO = [0] * back_CO
    TERM_THC = [0] * back_THC
    TERM_NMHC = [0] * back_NMHC
    TERM_WD_HR = [0] * back_WD_HR
    TERM_WIND_DIREC = [0] * back_WIND_DIREC
    TERM_WIND_SPEED = [0] * back_WIND_SPEED
    TERM_WS_HR = [0] * back_WS_HR
    TERM_AMB_TEMP = [0] * back_AMB_TEMP
    TERM_CH4 = [0] * back_CH4
    TERM_RAINFALL = [0] * back_RAINFALL
    for s in range(len(table[9]) - back_max):
        _y = float(table[9][s + back_max])
        y = b
        for k in range(back):
            data[back - 1 - k] = float(table[9][s + (back_max - back) + k])
        for n in range(back):
            y += data[n] * term[n]
        
        for k in range(back_2):
            data_2[back_2 - 1 - k] = float(table[9][s + (back_max - back_2) + k])
        for n in range(back_2):
            y += (data_2[n]**2) * term_2[n]


        for k in range(back_10):
            data_10[back_10 - 1 - k] = float(table[9-1][s + (back_max - back_10) + k])
        for n in range(back_10):
            y += data_10[n] * term_10[n]
        
        for k in range(back_10_2):
            data_10_2[back_10_2 - 1 - k] = float(table[9-1][s + (back_max - back_10_2) + k])
        for n in range(back_10_2):
            y += (data_10_2[n] ** 2) * term_10_2[n]

        for k in range(back_RH):
            data_RH[back_RH - 1 - k] = float(table[9+2][s + (back_max - back_RH) + k])
        for n in range(back_RH):
            y += data_RH[n] * term_RH[n]
        
        for k in range(back_NO):
            data_NO[back_NO - 1 - k] = float(table[9-5][s + (back_max - back_NO) + k])
        for n in range(back_NO):
            y += data_NO[n] * term_NO[n]
        
        for k in range(back_NO2):
            data_NO2[back_NO2 - 1 - k] = float(table[9-4][s + (back_max - back_NO2) + k])
        for n in range(back_NO2):
            y += data_NO2[n] * term_NO2[n]

        for k in range(back_NOx):
            data_NOx[back_NOx - 1 - k] = float(table[9-3][s + (back_max - back_NOx) + k])
        for n in range(back_NOx):
            y += data_NOx[n] * term_NOx[n]
        
        for k in range(back_O3):
            data_O3[back_O3 - 1 - k] = float(table[9-2][s + (back_max - back_O3) + k])
        for n in range(back_O3):
            y += data_O3[n] * term_O3[n]
        
        for k in range(back_SO2):
            data_SO2[back_SO2 - 1 - k] = float(table[9+3][s + (back_max - back_SO2) + k])
        for n in range(back_SO2):
            y += data_SO2[n] * term_SO2[n]
        
        for k in range(back_CO):
            data_CO[back_CO - 1 - k] = float(table[9-7][s + (back_max - back_CO) + k])
        for n in range(back_CO):
            y += data_CO[n] * term_CO[n]
        
        for k in range(back_THC):
            data_THC[back_THC - 1 - k] = float(table[9+4][s + (back_max - back_THC) + k])
        for n in range(back_THC):
            y += data_THC[n] * term_THC[n]
        
        for k in range(back_NMHC):
            data_NMHC[back_NMHC - 1 - k] = float(table[9-6][s + (back_max - back_NMHC) + k])
        for n in range(back_NMHC):
            y += data_NMHC[n] * term_NMHC[n]
        
        for k in range(back_WD_HR):
            data_WD_HR[back_WD_HR - 1 - k] = float(table[9+5][s + (back_max - back_WD_HR) + k])
        for n in range(back_WD_HR):
            y += data_WD_HR[n] * term_WD_HR[n]
        
        for k in range(back_WIND_DIREC):
            data_WIND_DIREC[back_WIND_DIREC - 1 - k] = float(table[9+6][s + (back_max - back_WIND_DIREC) + k])
        for n in range(back_WIND_DIREC):
            y += data_WIND_DIREC[n] * term_WIND_DIREC[n]
        
        for k in range(back_WIND_SPEED):
            data_WIND_SPEED[back_WIND_SPEED - 1 - k] = float(table[9+7][s + (back_max - back_WIND_SPEED) + k])
        for n in range(back_WIND_SPEED):
            y += data_WIND_SPEED[n] * term_WIND_SPEED[n]
        
        for k in range(back_WS_HR):
            data_WS_HR[back_WS_HR - 1 - k] = float(table[9+8][s + (back_max - back_WS_HR) + k])
        for n in range(back_WS_HR):
            y += data_WS_HR[n] * term_WS_HR[n]
        
        for k in range(back_AMB_TEMP):
            data_AMB_TEMP[back_AMB_TEMP - 1 - k] = float(table[9-9][s + (back_max - back_AMB_TEMP) + k])
        for n in range(back_AMB_TEMP):
            y += data_AMB_TEMP[n] * term_AMB_TEMP[n]
        
        for k in range(back_CH4):
            data_CH4[back_CH4 - 1 - k] = float(table[9-8][s + (back_max - back_CH4) + k])
        for n in range(back_CH4):
            y += data_CH4[n] * term_CH4[n]
        
        for k in range(back_RAINFALL):
            data_RAINFALL[back_RAINFALL - 1 - k] = float(table[9+1][s + (back_max - back_RAINFALL) + k])
        for n in range(back_RAINFALL):
            y += data_RAINFALL[n] * term_RAINFALL[n]

        L += (_y - y) ** 2
        count += 1
        B += (-2 * (_y - y))
        for n in range(back):
            TERM[n] += 2 * (_y - y) * (-data[n])
        for n in range(back_2):
            TERM_2[n] += 2 * (_y - y) * (-(data_2[n] ** 2))
        for n in range(back_10_2):
            TERM_10_2[n] += 2 * (_y - y) * (-(data_10_2[n] ** 2))
        for n in range(back_10):
            TERM_10[n] += 2 * (_y - y) * (-data_10[n])
        for n in range(back_RH):
            TERM_RH[n] += 2 * (_y - y) * (-data_RH[n])
        for n in range(back_NO):
            TERM_NO[n] += 2 * (_y - y) * (-data_NO[n])
        for n in range(back_NO2):
            TERM_NO2[n] += 2 * (_y - y) * (-data_NO2[n])
        for n in range(back_NOx):
            TERM_NOx[n] += 2 * (_y - y) * (-data_NOx[n])
        for n in range(back_O3):
            TERM_O3[n] += 2 * (_y - y) * (-data_O3[n])
        for n in range(back_SO2):
            TERM_SO2[n] += 2 * (_y - y) * (-data_SO2[n])
        for n in range(back_CO):
            TERM_CO[n] += 2 * (_y - y) * (-data_CO[n])
        for n in range(back_THC):
            TERM_THC[n] += 2 * (_y - y) * (-data_THC[n])
        for n in range(back_NMHC):
            TERM_NMHC[n] += 2 * (_y - y) * (-data_NMHC[n])
        for n in range(back_WD_HR):
            TERM_WD_HR[n] += 2 * (_y - y) * (-data_WD_HR[n])
        for n in range(back_WIND_DIREC):
            TERM_WIND_DIREC[n] += 2 * (_y - y) * (-data_WIND_DIREC[n])
        for n in range(back_WIND_SPEED):
            TERM_WIND_SPEED[n] += 2 * (_y - y) * (-data_WIND_SPEED[n])
        for n in range(back_WS_HR):
            TERM_WS_HR[n] += 2 * (_y - y) * (-data_WS_HR[n])
        for n in range(back_AMB_TEMP):
            TERM_AMB_TEMP[n] += 2 * (_y - y) * (-data_AMB_TEMP[n])
        for n in range(back_CH4):
            TERM_CH4[n] += 2 * (_y - y) * (-data_CH4[n])
        for n in range(back_RAINFALL):
            TERM_RAINFALL[n] += 2 * (_y - y) * (-data_RAINFALL[n])
    if back_10 != 0:
        G = 0
        for i in term_10:
            G += i
        print("PM10: %.4f" % G)
    if back_NO != 0:
        G = 0
        for i in term_NO:
            G += i
        print("NO: %.4f"% G)
    if back_NO2 != 0:
        G = 0
        for i in term_NO2:
            G += i
        print("NO2: %.4f"% G)
    if back_NOx != 0:
        G = 0
        for i in term_NOx:
            G += i
        print("NOx: %.4f"% G)
    if back_RH != 0:
        G = 0
        for i in term_RH:
            G += i
        print("RH: %.4f"% G)
    if back_O3 != 0:
        G = 0
        for i in term_O3:
            G += i
        print("O3: %.4f"% G)
    if back_CO != 0:
        G = 0
        for i in term_CO:
            G += i
        print("CO: %.4f"% G)
    if back_SO2 != 0:
        G = 0
        for i in term_SO2:
            G += i
        print("SO2: %.4f"% G)
    if back_THC != 0:
        G = 0
        for i in term_THC:
            G += i
        print("THC: %.4f"% G)
    if back_NMHC != 0:
        G = 0
        for i in term_NMHC:
            G += i
        print("NMHC: %.4f"% G)
    if back_WD_HR != 0:
        G = 0
        for i in term_WD_HR:
            G += i
        print("WD_HR: %.4f"% G)
    if back_WIND_DIREC != 0:
        G = 0
        for i in term_WIND_DIREC:
            G += i
        print("WIND_DIREC: %.4f"% G)
    if back_WIND_SPEED != 0:
        G = 0
        for i in term_WIND_SPEED:
            G += i
        print("WIND_SPEED: %.4f"% G)
    if back_WS_HR != 0:
        G = 0
        for i in term_WS_HR:
            G += i
        print("WS_HR: %.4f"% G)
    if back_AMB_TEMP != 0:
        G = 0
        for i in term_AMB_TEMP:
            G += i
        print("AMB_TEMP: %.4f"% G)
    if back_CH4 != 0:
        G = 0
        for i in term_CH4:
            G += i
        print("CH4: %.4f"% G)
    if back_RAINFALL != 0:
        G = 0
        for i in term_RAINFALL:
            G += i
        print("RAINFALL: %.4f"% G)
    print(z, L, (L/count)**0.5)
    print("\n")

    nl = L
    if (nl -ol)**2 < stop_point:
        break
    ol = nl

    historical_grad_B += B ** 2
    adjust_grad_B = B/(fudge_factor + historical_grad_B ** 0.5)
    b = b - master_stepsize * adjust_grad_B
    adjust_grad = [0] * back
    adjust_grad_2 = [0] * back_2
    adjust_grad_10_2 = [0] * back_10_2
    adjust_grad_10 = [0] * back_10
    adjust_grad_RH = [0] * back_RH
    adjust_grad_NO = [0] * back_NO
    adjust_grad_NO2 = [0] * back_NO2
    adjust_grad_NOx = [0] * back_NOx
    adjust_grad_O3 = [0] * back_O3
    adjust_grad_SO2 = [0] * back_SO2
    adjust_grad_CO = [0] * back_CO
    adjust_grad_THC = [0] * back_THC
    adjust_grad_NMHC = [0] * back_NMHC
    adjust_grad_WD_HR = [0] * back_WD_HR
    adjust_grad_WIND_DIREC = [0] * back_WIND_DIREC
    adjust_grad_WIND_SPEED = [0] * back_WIND_SPEED
    adjust_grad_WS_HR = [0] * back_WS_HR
    adjust_grad_AMB_TEMP = [0] * back_AMB_TEMP
    adjust_grad_CH4 = [0] * back_CH4
    adjust_grad_RAINFALL = [0] * back_RAINFALL

    for j in range(back):
        if grad[j] == 0:
            grad[j] = TERM[j] **2
        else:
            grad[j] += autocorr * grad[j] + (1- autocorr) * TERM[j] ** 2
        adjust_grad[j] = TERM[j]/(fudge_factor + grad[j] ** 0.5)
        m = term[j]
        term[j] = term[j] - master_stepsize * adjust_grad[j]
        term[j] += alpha * m
    
    for j in range(back_2):
        if grad_2[j] == 0:
            grad_2[j] = TERM_2[j] **2
        else:
            grad_2[j] += autocorr * grad_2[j] + (1- autocorr) * TERM_2[j] ** 2
        adjust_grad_2[j] = TERM_2[j]/(fudge_factor + grad_2[j] ** 0.5)
        m = term_2[j]
        term_2[j] = term_2[j] - master_stepsize * adjust_grad_2[j]
        term_2[j] += alpha * m
    
    for j in range(back_10):
        if grad_10[j] == 0:
            grad_10[j] = TERM_10[j] **2
        else:
            grad_10[j] += autocorr * grad_10[j] + (1- autocorr) * TERM_10[j] ** 2
        adjust_grad_10[j] = TERM_10[j]/(fudge_factor + grad_10[j] ** 0.5)
        m = term_10[j]
        term_10[j] = term_10[j] - master_stepsize * adjust_grad_10[j]
        term_10[j] += alpha * m
    
    for j in range(back_10_2):
        if grad_10_2[j] == 0:
            grad_10_2[j] = TERM_10_2[j] **2
        else:
            grad_10_2[j] += autocorr * grad_10_2[j] + (1- autocorr) * TERM_10_2[j] ** 2
        adjust_grad_10_2[j] = TERM_10_2[j]/(fudge_factor + grad_10_2[j] ** 0.5)
        m = term_10_2[j]
        term_10_2[j] = term_10_2[j] - master_stepsize * adjust_grad_10_2[j]
        term_10_2[j] += alpha * m
    
    for j in range(back_RH):
        if grad_RH[j] == 0:
            grad_RH[j] = TERM_RH[j] **2
        else:
            grad_RH[j] += autocorr * grad_RH[j] + (1- autocorr) * TERM_RH[j] ** 2
        adjust_grad_RH[j] = TERM_RH[j]/(fudge_factor + grad_RH[j] ** 0.5)
        m = term_RH[j]
        term_RH[j] = term_RH[j] - master_stepsize * adjust_grad_RH[j]
        term_RH[j] += alpha * m
    
    for j in range(back_NO):
        if grad_NO[j] == 0:
            grad_NO[j] = TERM_NO[j] **2
        else:
            grad_NO[j] += autocorr * grad_NO[j] + (1- autocorr) * TERM_NO[j] ** 2
        adjust_grad_NO[j] = TERM_NO[j]/(fudge_factor + grad_NO[j] ** 0.5)
        m = term_NO[j]
        term_NO[j] = term_NO[j] - master_stepsize * adjust_grad_NO[j]
        term_NO[j] += alpha * m
    
    for j in range(back_NO2):
        if grad_NO2[j] == 0:
            grad_NO2[j] = TERM_NO2[j] **2
        else:
            grad_NO2[j] += autocorr * grad_NO2[j] + (1- autocorr) * TERM_NO2[j] ** 2
        adjust_grad_NO2[j] = TERM_NO2[j]/(fudge_factor + grad_NO2[j] ** 0.5)
        m = term_NO2[j]
        term_NO2[j] = term_NO2[j] - master_stepsize * adjust_grad_NO2[j]
        term_NO2[j] += alpha * m
    
    for j in range(back_NOx):
        if grad_NOx[j] == 0:
            grad_NOx[j] = TERM_NOx[j] **2
        else:
            grad_NOx[j] += autocorr * grad_NOx[j] + (1- autocorr) * TERM_NOx[j] ** 2
        adjust_grad_NOx[j] = TERM_NOx[j]/(fudge_factor + grad_NOx[j] ** 0.5)
        m = term_NOx[j]
        term_NOx[j] = term_NOx[j] - master_stepsize * adjust_grad_NOx[j]
        term_NOx[j] += alpha * m
    
    for j in range(back_O3):
        if grad_O3[j] == 0:
            grad_O3[j] = TERM_O3[j] **2
        else:
            grad_O3[j] += autocorr * grad_O3[j] + (1- autocorr) * TERM_O3[j] ** 2
        adjust_grad_O3[j] = TERM_O3[j]/(fudge_factor + grad_O3[j] ** 0.5)
        m = term_O3[j]
        term_O3[j] = term_O3[j] - master_stepsize * adjust_grad_O3[j]
        term_O3[j] += alpha * m
    
    for j in range(back_SO2):
        if grad_SO2[j] == 0:
            grad_SO2[j] = TERM_SO2[j] **2
        else:
            grad_SO2[j] += autocorr * grad_SO2[j] + (1- autocorr) * TERM_SO2[j] ** 2
        adjust_grad_SO2[j] = TERM_SO2[j]/(fudge_factor + grad_SO2[j] ** 0.5)
        m = term_SO2[j]
        term_SO2[j] = term_SO2[j] - master_stepsize * adjust_grad_SO2[j]
        term_SO2[j] += alpha * m
    
    for j in range(back_CO):
        if grad_CO[j] == 0:
            grad_CO[j] = TERM_CO[j] **2
        else:
            grad_CO[j] += autocorr * grad_CO[j] + (1- autocorr) * TERM_CO[j] ** 2
        adjust_grad_CO[j] = TERM_CO[j]/(fudge_factor + grad_CO[j] ** 0.5)
        m = term_CO[j]
        term_CO[j] = term_CO[j] - master_stepsize * adjust_grad_CO[j]
        term_CO[j] += alpha * m
    
    for j in range(back_THC):
        if grad_THC[j] == 0:
            grad_THC[j] = TERM_THC[j] **2
        else:
            grad_THC[j] += autocorr * grad_THC[j] + (1- autocorr) * TERM_THC[j] ** 2
        adjust_grad_THC[j] = TERM_THC[j]/(fudge_factor + grad_THC[j] ** 0.5)
        m = term_THC[j]
        term_THC[j] = term_THC[j] - master_stepsize * adjust_grad_THC[j]
        term_THC[j] += alpha * m
    
    for j in range(back_NMHC):
        if grad_NMHC[j] == 0:
            grad_NMHC[j] = TERM_NMHC[j] **2
        else:
            grad_NMHC[j] += autocorr * grad_NMHC[j] + (1- autocorr) * TERM_NMHC[j] ** 2
        adjust_grad_NMHC[j] = TERM_NMHC[j]/(fudge_factor + grad_NMHC[j] ** 0.5)
        m = term_NMHC[j]
        term_NMHC[j] = term_NMHC[j] - master_stepsize * adjust_grad_NMHC[j]
        term_NMHC[j] += alpha * m
    
    for j in range(back_WD_HR):
        if grad_WD_HR[j] == 0:
            grad_WD_HR[j] = TERM_WD_HR[j] **2
        else:
            grad_WD_HR[j] += autocorr * grad_WD_HR[j] + (1- autocorr) * TERM_WD_HR[j] ** 2
        adjust_grad_WD_HR[j] = TERM_WD_HR[j]/(fudge_factor + grad_WD_HR[j] ** 0.5)
        m = term_WD_HR[j]
        term_WD_HR[j] = term_WD_HR[j] - master_stepsize * adjust_grad_WD_HR[j]
        term_WD_HR[j] += alpha * m
    
    for j in range(back_WIND_DIREC):
        if grad_WIND_DIREC[j] == 0:
            grad_WIND_DIREC[j] = TERM_WIND_DIREC[j] **2
        else:
            grad_WIND_DIREC[j] += autocorr * grad_WIND_DIREC[j] + (1- autocorr) * TERM_WIND_DIREC[j] ** 2
        adjust_grad_WIND_DIREC[j] = TERM_WIND_DIREC[j]/(fudge_factor + grad_WIND_DIREC[j] ** 0.5)
        m = term_WIND_DIREC[j]
        term_WIND_DIREC[j] = term_WIND_DIREC[j] - master_stepsize * adjust_grad_WIND_DIREC[j]
        term_WIND_DIREC[j] += alpha * m
    
    for j in range(back_WIND_SPEED):
        if grad_WIND_SPEED[j] == 0:
            grad_WIND_SPEED[j] = TERM_WIND_SPEED[j] **2
        else:
            grad_WIND_SPEED[j] += autocorr * grad_WIND_SPEED[j] + (1- autocorr) * TERM_WIND_SPEED[j] ** 2
        adjust_grad_WIND_SPEED[j] = TERM_WIND_SPEED[j]/(fudge_factor + grad_WIND_SPEED[j] ** 0.5)
        m = term_WIND_SPEED[j]
        term_WIND_SPEED[j] = term_WIND_SPEED[j] - master_stepsize * adjust_grad_WIND_SPEED[j]
        term_WIND_SPEED[j] += alpha * m
    
    for j in range(back_WS_HR):
        if grad_WS_HR[j] == 0:
            grad_WS_HR[j] = TERM_WS_HR[j] **2
        else:
            grad_WS_HR[j] += autocorr * grad_WS_HR[j] + (1- autocorr) * TERM_WS_HR[j] ** 2
        adjust_grad_WS_HR[j] = TERM_WS_HR[j]/(fudge_factor + grad_WS_HR[j] ** 0.5)
        m = term_WS_HR[j]
        term_WS_HR[j] = term_WS_HR[j] - master_stepsize * adjust_grad_WS_HR[j]
        term_WS_HR[j] += alpha * m

    for j in range(back_AMB_TEMP):
        if grad_AMB_TEMP[j] == 0:
            grad_AMB_TEMP[j] = TERM_AMB_TEMP[j] **2
        else:
            grad_AMB_TEMP[j] += autocorr * grad_AMB_TEMP[j] + (1- autocorr) * TERM_AMB_TEMP[j] ** 2
        adjust_grad_AMB_TEMP[j] = TERM_AMB_TEMP[j]/(fudge_factor + grad_AMB_TEMP[j] ** 0.5)
        m = term_AMB_TEMP[j]
        term_AMB_TEMP[j] = term_AMB_TEMP[j] - master_stepsize * adjust_grad_AMB_TEMP[j]
        term_AMB_TEMP[j] += alpha * m
    
    for j in range(back_CH4):
        if grad_CH4[j] == 0:
            grad_CH4[j] = TERM_CH4[j] **2
        else:
            grad_CH4[j] += autocorr * grad_CH4[j] + (1- autocorr) * TERM_CH4[j] ** 2
        adjust_grad_CH4[j] = TERM_CH4[j]/(fudge_factor + grad_CH4[j] ** 0.5)
        m = term_CH4[j]
        term_CH4[j] = term_CH4[j] - master_stepsize * adjust_grad_CH4[j]
        term_CH4[j] += alpha * m
    
    for j in range(back_RAINFALL):
        if grad_RAINFALL[j] == 0:
            grad_RAINFALL[j] = TERM_RAINFALL[j] **2
        else:
            grad_RAINFALL[j] += autocorr * grad_RAINFALL[j] + (1- autocorr) * TERM_RAINFALL[j] ** 2
        adjust_grad_RAINFALL[j] = TERM_RAINFALL[j]/(fudge_factor + grad_RAINFALL[j] ** 0.5)
        m = term_RAINFALL[j]
        term_RAINFALL[j] = term_RAINFALL[j] - master_stepsize * adjust_grad_RAINFALL[j]
        term_RAINFALL[j] += alpha * m
#######################################################
ofile.write("id,value")
ofile.write('\n')

Test_table = []
for line in test:
    tokens = line.split(',')
    if (tokens[1] == 'RAINFALL'):
        for i in range(len(tokens)):
            if tokens[i] == 'NR':
                tokens[i] = 0
            elif tokens[i] == 'NR\r\n':
                tokens[i] = 0
            elif tokens[i] == 'NR\n':
                tokens[i] = 0
    Test_table.append(tokens)

for i in range(9, len(Test_table), 18):
    y = b
    for s in range(back):
        test_data[s] = float(Test_table[i][s + 11 - back])
    for n in range(back):
        y += test_data[n] * term[back - 1 -n]
    
    for s in range(back_2):
        test_data_2[s] = float(Test_table[i][s + 11 - back_2])
    for n in range(back_2):
        y += (test_data_2[n] ** 2) * term_2[back_2 - 1 - n]

    for s in range(back_10):
        test_data_10[s] = float(Test_table[i-1][s + 11 - back_10])
    for n in range(back_10):
        y += test_data_10[n] * term_10[back_10 - 1 -n]
    
    for s in range(back_10_2):
        test_data_10_2[s] = float(Test_table[i-1][s + 11 - back_10_2])
    for n in range(back_10_2):
        y += (test_data_10_2[n] ** 2) * term_10_2[back_10_2 - 1 - n]


    for s in range(back_RH):
        test_data_RH[s] = float(Test_table[i+2][s + 11 - back_RH])
    for n in range(back_RH):
        y += test_data_RH[n] * term_RH[back_RH - 1 -n]
    
    for s in range(back_NO):
        test_data_NO[s] = float(Test_table[i-5][s + 11 - back_NO])
    for n in range(back_NO):
        y += test_data_NO[n] * term_NO[back_NO - 1 -n]

    for s in range(back_NO2):
        test_data_NO2[s] = float(Test_table[i-4][s + 11 - back_NO2])
    for n in range(back_NO2):
        y += test_data_NO2[n] * term_NO2[back_NO2 - 1 -n]

    for s in range(back_NOx):
        test_data_NOx[s] = float(Test_table[i-3][s + 11 -  back_NOx])
    for n in range(back_NOx):
        y += test_data_NOx[n] * term_NOx[back_NOx - 1 -n]
    
    for s in range(back_O3):
        test_data_O3[s] = float(Test_table[i-2][s + 11 - back_O3])
    for n in range(back_O3):
        y += test_data_O3[n] * term_O3[back_O3 - 1 -n]

    for s in range(back_SO2):
        test_data_SO2[s] = float(Test_table[i+3][s + 11 - back_SO2])
    for n in range(back_SO2):
        y += test_data_SO2[n] * term_SO2[back_SO2 - 1 -n]
    
    for s in range(back_CO):
        test_data_CO[s] = float(Test_table[i-7][s + 11 - back_CO])
    for n in range(back_CO):
        y += test_data_CO[n] * term_CO[back_CO - 1 -n]
    
    for s in range(back_THC):
        test_data_THC[s] = float(Test_table[i+4][s + 11 - back_THC])
    for n in range(back_THC):
        y += test_data_THC[n] * term_THC[back_THC - 1 -n]
    
    for s in range(back_NMHC):
        test_data_NMHC[s] = float(Test_table[i-6][s + 11 - back_NMHC])
    for n in range(back_NMHC):
        y += test_data_NMHC[n] * term_NMHC[back_NMHC - 1 -n]
    
    for s in range(back_WD_HR):
        test_data_WD_HR[s] = float(Test_table[i+5][s + 11 - back_WD_HR])
    for n in range(back_WD_HR):
        y += test_data_WD_HR[n] * term_WD_HR[back_WD_HR - 1 -n]
    
    for s in range(back_WIND_DIREC):
        test_data_WIND_DIREC[s] = float(Test_table[i+6][s + 11 - back_WIND_DIREC])
    for n in range(back_WIND_DIREC):
        y += test_data_WIND_DIREC[n] * term_WIND_DIREC[back_WIND_DIREC - 1 -n]
    
    for s in range(back_WIND_SPEED):
        test_data_WIND_SPEED[s] = float(Test_table[i+7][s + 11 - back_WIND_SPEED])
    for n in range(back_WIND_SPEED):
        y += test_data_WIND_SPEED[n] * term_WIND_SPEED[back_WIND_SPEED - 1 -n]
    
    for s in range(back_WS_HR):
        test_data_WS_HR[s] = float(Test_table[i+8][s + 11 - back_WS_HR])
    for n in range(back_WS_HR):
        y += test_data_WS_HR[n] * term_WS_HR[back_WS_HR - 1 -n]
    
    for s in range(back_AMB_TEMP):
        test_data_AMB_TEMP[s] = float(Test_table[i-9][s + 11 - back_AMB_TEMP])
    for n in range(back_AMB_TEMP):
        y += test_data_AMB_TEMP[n] * term_AMB_TEMP[back_AMB_TEMP - 1 -n]
    
    for s in range(back_CH4):
        test_data_CH4[s] = float(Test_table[i-8][s + 11 - back_CH4])
    for n in range(back_CH4):
        y += test_data_CH4[n] * term_CH4[back_CH4 - 1 -n]
    
    for s in range(back_RAINFALL):
        test_data_RAINFALL[s] = float(Test_table[i+1][s + 11 - back_RAINFALL])
    for n in range(back_RAINFALL):
        y += test_data_RAINFALL[n] * term_RAINFALL[back_RAINFALL - 1 -n]

    ofile.write(Test_table[i][0] + ',')
    ofile.write(str(y))
    ofile.write('\n')
ofile.close()
test.close()

