nb_classes = 10
batch_size = 128
nb_epoch = 250
encoding_dim = 256
add_size = 5000

# encoder
model = Sequential()
model.add(Dense(encoding_dim, activation = 'relu', input_shape = (3072,)))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(encoding_dim, activation = 'relu'))
model.add(Dense(3072, activation = 'linear'))

model.compile(loss = 'mse', optimizer = 'rmsprop', metrics = ['accuracy'])

model.fit(x_train, x_train,
                batch_size = batch_size,
                nb_epoch = nb_epoch,
                verbose = 1,
                validation_data = (x_test, x_test))

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

