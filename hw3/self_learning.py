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

#os.environ["THEANO_FLAGS"] = "device=gpu0"


batch_size = 32
nb_classes = 10
nb_epoch = 250
data_augmentation = True
iteration = 5
threshold = 0.98

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3
input_shape = (img_channels, img_rows, img_cols)

data_repo = sys.argv[1]
model_name = sys.argv[2]

########## load raw data #########################################
def load_data(sig):
    if sig == 'l':
        print "Loading data from all_label.p ..."
        return pickle.load(open(data_repo + "/all_label.p", "rb"))
    elif sig == 'ul':
        print "Loading data from all_unlabel.p ..."
        return pickle.load(open(data_repo + "/all_unlabel.p", "rb"))
    elif sig == 't':
        print "Loading data from test.p ..."
        return pickle.load(open(data_repo + "/test.p", "rb"))

l_data = load_data('l')
ul_data = load_data('ul')

x_train = []
y_train = []
x_ul = []


# initialize x_train, y_train, x_test
for i in range(nb_classes):
    temp = [0] * 10
    temp[i] = 1
    for j in range(len(l_data[i])):
        x_train.append(l_data[i][j])
        y_train.append(temp)
for i in range(len(ul_data)):
    x_ul.append(ul_data[i])



x_train = np.array(x_train)
y_train = np.array(y_train)
x_ul = np.array(x_ul)

if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_ul = x_ul.reshape(x_ul.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_ul = x_ul.reshape(x_ul.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_ul = x_ul.astype('float32')

x_train /= 255
x_ul /= 255

print 'x_train shape: ', x_train.shape
print x_train.shape[0], 'train samples'
print x_ul.shape[0], 'unlabel samples'


# define model
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(Dense(output_dim = 256))
model.add(Activation('relu'))
model.add(Dense(output_dim = 128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(output_dim = nb_classes))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

if data_augmentation == True:
    print "Data augmentation..."
    datagen = ImageDataGenerator(
            featurewise_center = False,  # set input mean to 0 over the dataset
            samplewise_center = False,  # set each sample mean to 0
            featurewise_std_normalization = False,  # divide inputs by std of the dataset
            samplewise_std_normalization = False,  # devide each input by its std
            zca_whitening = False,  # apply ZCA whitening
            rotation_range = 12,  # randomly rotate images in range
            width_shift_range = 0.1,  # randomly shift images horizontally
            height_shift_range = 0.1,  # randomly shift images vertically
            horizontal_flip = True,  # randomly flip images
            vertical_flip = False)  # randomly flip images
    datagen.fit(x_train)
else:
    print 'No data augmentation...'

for i in range(iteration):
    if i > 0:
        nb_epoch = 100
    if data_augmentation is True:
        model.fit_generator(datagen.flow(x_train, y_train,
                            batch_size = batch_size),
                            samples_per_epoch = x_train.shape[0],
                            nb_epoch = nb_epoch)
    else:
        model.fit(x_train, y_train,
                batch_size = batch_size,
                nb_epoch = nb_epoch,
                shuffle = True)
    r = model.predict(x_ul)
    tmp_x = []
    tmp_y = []
    t = []
    for j in range(len(r)):
        m, idx = 0, 0
        for k in range(len(r[j])):
            if r[j][k] > m:
                m = r[j][k]
                idx = k
        if m > threshold:
            temp = [0] * nb_classes
            temp[idx] = 1
            tmp_x.append(x_ul[j])
            tmp_y.append(temp)
            t.append(j)
    print "x_ul shape", x_ul.shape
    if len(tmp_x) > 0 and len(tmp_y) > 0:
        tmp_x = np.array(tmp_x)
        tmp_y = np.array(tmp_y)
        print "tmp_x shape", tmp_x.shape
        print "tmp_y shape", tmp_y.shape
        x_train = np.concatenate((x_train, tmp_x), axis = 0)
        y_train = np.concatenate((y_train, tmp_y), axis = 0)
    print "x_train shape", x_train.shape
    print "y_train shape", y_train.shape
    x_ul = np.delete(x_ul, t, axis = 0)
    print "x_ul shape", x_ul.shape

model.save(model_name)
'''
result = model.predict(x_test)
out = []
for i in range(len(result)):
    m , idx = 0, 0
    for j in range(len(result[i])):
        if result[i][j] > m:
            m = result[i][j]
            idx = j
    out.append(idx)
ofile = open("prediction.csv", "wb")
ofile.write("ID,class")
ofile.write('\n')
for i in range(len(out)):
    ofile.write(str(i) + ',')
    ofile.write(str(out[i]))
    ofile.write('\n')
'''

