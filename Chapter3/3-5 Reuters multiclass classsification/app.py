import numpy
import keras

import draw


def one_hot_encoding(sequences, dimension=10000):
    results = numpy.zeros((len(sequences), dimension))
    for index, sequence in enumerate(sequences):
        results[index, sequence] = 1.
    return results


(train_data, train_label), (test_data, test_label) = keras.datasets.reuters.load_data(num_words=10000)

word_index = keras.datasets.reuters.get_word_index()
reverse_word_index = dict([(value, key) for key, value in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_newswire)

x_train_data = one_hot_encoding(train_data)
y_train_label = keras.utils.np_utils.to_categorical(train_label)

x_test_data = one_hot_encoding(test_data)
x_test_label = keras.utils.np_utils.to_categorical(test_label)


model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

x_validation = x_train_data[:1000]
partial_x_train = x_train_data[1000:]

y_validation = y_train_label[:1000]
partial_y_train = y_train_label[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_validation, y_validation))
draw.train_and_epochs(history)
