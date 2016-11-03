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

#os.environ["THEANO_FLAGS"] = "device=gpu0"


batch_size = 50
nb_classes = 10
nb_epoch = 50
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3
input_shape = (img_channels, img_rows, img_cols)

data_repo = sys.argv[1]

########## load raw data #########################################
def load_label_data():
    print "Loading data from all_label.p ..."
    l = pickle.load(open(data_repo + "all_label.p", "rb"))
    return l

def load_unlabel_data():
    print "Loading data from all_unlabel.p ..."
    l = pickle.load(open(data_repo + "all_unlabel.p", "rb"))
    return l

def load_test_data():
    print "Loading data from test.p ..."
    l = pickle.load(open(data_repo + "test.p", "rb"))
    return l

l_data = load_label_data()
#ul_data = load_unlabel_data()
test_data = load_test_data()

x_train = []
y_train = []
x_ul = []
x_test = []


# initialize x_train, y_train, x_test
for i in range(nb_classes):
    temp = [0] * 10
    temp[i] = 1
    for j in range(len(l_data[i])):
        x_train.append(l_data[i][j])
        y_train.append(temp)


for i in test_data['data']:
        x_test.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)

if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols) 
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print 'x_train shape: ', x_train.shape
print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'


# define model
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape = input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

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
else:
    print "No data augmentation..."
    model.fit(x_train, y_train,
            batch_size = batch_size,
            nb_epoch = nb_epoch,
            shuffle = True)

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


