#! /usr/bin/env python
import sys
import os
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import *
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model

data_repo = sys.argv[1]
model_name = sys.argv[2]

def load_data(sig):
    if sig == 'l':
        print "Loading data from all_label.p ..."
        return pickle.load(open(data_repo + "all_label.p", "rb"))
    elif sig == 'ul':
        print "Loading data from all_unlabel.p ..."
        return pickle.load(open(data_repo + "all_unlabel.p", "rb"))
    elif sig == 't':
        print "Loading data from test.p ..."
        return pickle.load(open(data_repo + "test.p", "rb"))

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3
input_shape = (img_channels, img_rows, img_cols)

data_repo = sys.argv[1]

test_data = load_data('t')
x_test = []
for i in test_data['data']:
        x_test.append(i)
x_test = np.array(x_test)
if K.image_dim_ordering() == 'th':
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_test = x_test.astype('float32')
x_test /= 255
print x_test.shape[0], 'test samples'

model = load_model(model_name)


result = model.predict(x_test)
out = []
for i in range(len(result)):
    m , idx = 0, 0
    for j in range(len(result[i])):
        if result[i][j] > m:
            m = result[i][j]
            idx = j
    out.append(idx)
ofile = open(sys.argv[3], "wb")
ofile.write("ID,class")
ofile.write('\n')
for i in range(len(out)):
    ofile.write(str(i) + ',')
    ofile.write(str(out[i]))
    ofile.write('\n')


