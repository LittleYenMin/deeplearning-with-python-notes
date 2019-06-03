import keras

# Load MNIST dataset from keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# create Sequential model
network = keras.models.Sequential()
# add 2 Dense Layer to the model
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(keras.layers.Dense(10, activation='softmax'))
# compile model by optimizer, loss and metrics
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# prepare data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# category the labels
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# train model by fit() method
history = network.fit(train_images, train_labels, epochs=5, batch_size=128)
print(history)

# review the train result by evaluate() method and test datasets.
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('loss: {} - acc: {}'.format(test_loss, test_acc))
