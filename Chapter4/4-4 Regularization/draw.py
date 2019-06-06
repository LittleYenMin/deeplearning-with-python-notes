import matplotlib.pyplot as plt


def loss_and_epochs(original_history, regularization_history):
    history_dict = original_history.history
    l2_history_dict = regularization_history.history
    l2_val_lost_values = l2_history_dict['val_loss']
    val_lost_values = history_dict['val_loss']
    epochs = range(1, len(val_lost_values)+1)
    plt.plot(epochs, val_lost_values, 'bo', label='Original model')
    plt.plot(epochs, l2_val_lost_values, 'b', label='L2-regularized model')
    plt.title('Original and Regularized validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation loss')
    plt.legend()
    plt.show()
