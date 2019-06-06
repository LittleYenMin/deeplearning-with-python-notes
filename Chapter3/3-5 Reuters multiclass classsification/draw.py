import matplotlib.pyplot as plt


def train_and_epochs(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_lost_values = history_dict['val_loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_lost_values, 'b', label='Validation loss')
    plt.title('Traning and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def acc_and_epochs(history):
    history_dict = history.history
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Traning and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
