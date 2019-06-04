import imdb
import draw


history = imdb.model.fit(imdb.partial_x_train, imdb.partial_y_train, batch_size=512, epochs=4, validation_data=(imdb.x_val, imdb.y_val))
draw.acc_and_epochs(history)
result = imdb.model.evaluate(imdb.x_test, imdb.y_test)
print('loss: {}, accuracy: {}'.format(result[0], result[1]))
