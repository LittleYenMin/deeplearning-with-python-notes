import example_3_4
import l2_regularizers
import imdb
import draw


original_history = example_3_4.model.fit(imdb.partial_x_train, imdb.partial_y_train, batch_size=512, epochs=20, validation_data=(imdb.x_val, imdb.y_val))
l2_regularized_history = l2_regularizers.model.fit(imdb.partial_x_train, imdb.partial_y_train, batch_size=512, epochs=20, validation_data=(imdb.x_val, imdb.y_val))
draw.loss_and_epochs(original_history, l2_regularized_history)
