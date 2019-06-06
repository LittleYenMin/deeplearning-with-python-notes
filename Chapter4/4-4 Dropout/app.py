import example_3_4
import dropout
import imdb
import draw


original_history = example_3_4.model.fit(imdb.partial_x_train, imdb.partial_y_train, batch_size=512, epochs=20, validation_data=(imdb.x_val, imdb.y_val))
dropout_history = dropout.model.fit(imdb.partial_x_train, imdb.partial_y_train, batch_size=512, epochs=20, validation_data=(imdb.x_val, imdb.y_val))
draw.loss_and_epochs(original_history, dropout_history)
