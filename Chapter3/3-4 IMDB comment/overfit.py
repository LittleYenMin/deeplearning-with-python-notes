import numpy
import keras

import draw


def vectorize_sequences(sequences, dimension: int = 10000):
    results = numpy.zeros(shape=(len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Load IMDB comments dataset from keras and constraint the number of the word in train data is top 10000
(train_datas, train_labels), (test_datas, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

reverse_word_dict = dict([(value, key) for (key, value) in keras.datasets.imdb.get_word_index().items()])

# when load_data from imdb dataset the index will be plus 3 by keras,
# 0~2 have special uses, it means we need minus 3 to map the dictionary.
print(' '.join([reverse_word_dict.get(word_index-3, '?') for word_index in train_datas[0]]))
x_train = vectorize_sequences(train_datas)
x_test = vectorize_sequences(test_datas)

y_train = numpy.asarray(train_labels).astype('float32')
y_test = numpy.asarray(test_labels).astype('float32')

model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))
draw.acc_and_epochs(history)
result = model.evaluate(x_test, y_test)
print('loss: {}, accuracy: {}'.format(result[0], result[1]))
