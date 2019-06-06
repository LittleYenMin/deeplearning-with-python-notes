import keras


model = keras.Sequential()
model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(10000, )))
model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.binary_accuracy])
