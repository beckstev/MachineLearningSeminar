import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np 
def plot_history_special(history, path):
    """function for plotting the loss and the accuracy from a saved model
    :param history: Dataframe generated from model_history.csv
    :param path: saving path
    """
    print("plot history")
    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)

    num_epochs = len(history['loss'])
    epoche = np.arange(1, num_epochs + 1)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 3.58), sharex=True)

    ax0.set_ylabel('Loss')
    ax0.plot(epoche, history['loss'], label='Training')
    ax0.plot(epoche, history['val_loss'], label='Validation')

    ymin, ymax = ax0.get_ylim()
    y_ticks = np.round(np.linspace(ymin, ymax, 4), 2)
    ax0.set_yticks(y_ticks)
    ax0.legend()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epoche, history['acc'])
    ax1.plot(epoche, history['val_acc'])

    ymin, ymax = ax1.get_ylim()
    y_ticks = np.round(np.linspace(ymin, ymax, 4), 2)
    ax1.set_yticks(y_ticks)
    plt.tight_layout()
    plt.savefig("{}/history.pdf".format(path), dpi=500,
                pad_inches=0, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    history = pd.read_csv('./model_history.csv')
    plot_history_special(history, '.')
