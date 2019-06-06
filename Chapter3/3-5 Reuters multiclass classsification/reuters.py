import numpy
import keras


def one_hot_encoding(sequences, dimension=10000):
    results = numpy.zeros((len(sequences), dimension))
    for index, sequence in enumerate(sequences):
        results[index, sequence] = 1.
    return results


word_index = keras.datasets.reuters.get_word_index()
reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def decoded_newswire(sequences):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


(train_data, train_label), (test_data, test_label) = keras.datasets.reuters.load_data(num_words=10000)

x_train_data = one_hot_encoding(train_data)
y_train_label = keras.utils.np_utils.to_categorical(train_label)

x_test_data = one_hot_encoding(test_data)
y_test_label = keras.utils.np_utils.to_categorical(test_label)


model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

partial_x_train = x_train_data[1000:]
x_validation = x_train_data[:1000]
partial_y_train = y_train_label[1000:]
y_validation = y_train_label[:1000]
