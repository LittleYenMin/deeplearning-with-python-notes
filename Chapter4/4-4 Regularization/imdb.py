import keras
import numpy


def vectorize_sequences(sequences, dimension: int = 10000):
    results = numpy.zeros(shape=(len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Load IMDB comments dataset from keras and constraint the number of the word in train data is top 10000
(train_datas, train_labels), (test_datas, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_datas)
x_test = vectorize_sequences(test_datas)

y_train = numpy.asarray(train_labels).astype('float32')
y_test = numpy.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
