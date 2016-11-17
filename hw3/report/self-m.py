iteration = 8
threshold = 0.98

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
    if len(tmp_x) > 0 and len(tmp_y) > 0:
        tmp_x = np.array(tmp_x)
        tmp_y = np.array(tmp_y)
        x_train = np.concatenate((x_train, tmp_x), axis = 0)
        y_train = np.concatenate((y_train, tmp_y), axis = 0)
    x_ul = np.delete(x_ul, t, axis = 0)

