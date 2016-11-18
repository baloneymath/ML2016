#! /usr/bin/env python
import sys
import os
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import *
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading, label_propagation
from sklearn.manifold import TSNE

data_repo = sys.argv[1]

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

nb_classes = 10
batch_size = 128
nb_epoch = 120
encoding_dim = 256
add_size = 5000

l_data = load_data('l')
ul_data = load_data('ul')
test_data = load_data('t')

x_test = []
x_train = []
y_train = []
x_ul = []
y_ul = []


for i in range(nb_classes):
    temp = [0] * 10
    temp[i] = 1
    for j in range(len(l_data[i])):
        x_train.append(l_data[i][j])
        y_train.append(temp)
for i in range(len(ul_data)):
    x_ul.append(ul_data[i])
    y_ul.append(-1)
for i in test_data['data']:
    x_test.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_ul = np.array(x_ul)
y_ul = np.array(y_ul)
x_test = np.array(x_test)

x_train = x_train.astype('float32') / 255
x_ul = x_ul.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = Sequential()
model.add(Dense(encoding_dim, activation = 'relu', input_shape = (3072,)))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(3072, activation = 'linear'))

model.compile(loss = 'mse', optimizer = 'rmsprop')

model.fit(x_train, x_train,
                batch_size = batch_size,
                nb_epoch = nb_epoch,
                verbose = 1,
                validation_data = (x_test, x_test))

# encoding
encoder = K.function([model.layers[0].input], [model.layers[2].output])
encoded_x_train = encoder([x_train])[0]

# calculate k-means
ave = []
for i in range(nb_classes):
    ave.append([0.0 for m in range(encoding_dim)])

for i in range(nb_classes):
    for idx in range(500):
        pos = i * 500 + idx
        for j in range(encoding_dim):
            ave[i][j] += encoded_x_train[pos][j]
    for k in range(encoding_dim):
        ave[i][k] /= 500
print 'phase 1'
encoded_ul = encoder([x_ul])[0]
c = []
for i in range(len(x_ul)):
    lb = -1
    m = 1e10
    for j in range(nb_classes):
        mse = 0.0
        for k in range(encoding_dim):
            mse += (encoded_ul[i][k] - ave[j][k]) ** 2
        if mse < m:
            lb = j
            m = mse
    c.append((i, lb, m))
c.sort(key = lambda x: x[2])
print 'phase 2'
new_x = []
new_y = []
for i in range(add_size):
    tmp_y = [0.] * nb_classes
    tmp_y[c[i][1]] = 1.
    new_x.append(x_ul[c[i][0]])
    new_y.append(tmp_y)

new_x = np.array(new_x)
new_y = np.array(new_y)

new_x = new_x.astype('float32') / 255

print 'y_train shape', y_train.shape
print 'new_y shape', new_y.shape

x_train = np.concatenate((x_train, new_x), axis = 0)
y_train = np.concatenate((y_train, new_y), axis = 0)
x_train = x_train.reshape(len(x_train), 3, 32, 32)

# define model
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape = (3, 32, 32)))
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
print "Data augmentation..."
datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # devide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in range
        width_shift_range = 0.1,  # randomly shift images horizontally
        height_shift_range = 0.1,  # randomly shift images vertically
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False)  # randomly flip images
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train,
                    batch_size = batch_size),
                    samples_per_epoch = x_train.shape[0],
                    nb_epoch = nb_epoch)

model.save(sys.argv[2])

