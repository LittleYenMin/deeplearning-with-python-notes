import numpy


import draw
import reuters

history = reuters.model.fit(reuters.partial_x_train, reuters.partial_y_train,
                            epochs=9, batch_size=512, validation_data=(reuters.x_validation, reuters.y_validation))
draw.train_and_epochs(history)
result = reuters.model.evaluate(reuters.x_test_data, reuters.y_test_label)
print('loss: {}, accuracy: {}'.format(result[0], result[1]))

predictions = reuters.model.predict(reuters.x_test_data)
print(predictions)
print('odds: ', predictions[0])
print('sum of all odds: {}'.format(numpy.sum(predictions[0])))
max_odds_index = numpy.argmax(predictions[0])
print('max odds of class: {}, odds: {}'.format(max_odds_index, float(predictions[0][max_odds_index])))
