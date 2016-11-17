if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols) 
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32') / 255
y_train = y_train.astype('float32')
x_test = x_test.astype('float32') / 255

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
                optimizer = 'adadelta',
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
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size = batch_size),
                        samples_per_epoch = x_train.shape[0],
                        nb_epoch = nb_epoch)

