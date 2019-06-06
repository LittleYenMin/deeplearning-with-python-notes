import matplotlib.pyplot as plt


def loss_and_epochs(original_history, history):
    history_dict = original_history.history
    dropout_history_dict = history.history
    dp_val_lost_values = dropout_history_dict['val_loss']
    val_lost_values = history_dict['val_loss']
    epochs = range(1, len(val_lost_values)+1)
    plt.plot(epochs, val_lost_values, 'bo', label='Original model')
    plt.plot(epochs, dp_val_lost_values, 'b', label='Dropout model')
    plt.title('Original and Dropout validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation loss')
    plt.legend()
    plt.show()
