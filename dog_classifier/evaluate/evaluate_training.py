import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from keras.callbacks import Callback


def plot_history(network_history, fname):
    """function for plotting the loss and the accuracy"""
    print("plot history")
    plt.figure()
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend([r'Training', r'Validation'])
    plt.savefig("build/{}_loss.pdf".format(fname))
    plt.clf()

    plt.figure()
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend([r'Training', r'Validation'], loc='lower right')
    plt.savefig("build/{}_acc.pdf".format(fname))
    plt.clf()


def prob_multiclass(Y_pred, Y_test, label, fname):
    """defines a multiclass probability in the case that the output of the cnn
    cannot be interpreted as a probability. Also plots for a given label"""
    print("plot multiclass probability")
    n_cls = len(Y_pred[0])

    Y_prob = []
    for i in range(len(Y_pred)):
        numerator = Y_pred[i, label]
        denominator = 0.0
        for idx in range(n_cls):
            denominator += Y_pred[i, idx]

        Y_prob.append(numerator/denominator)

    Y_pred_prob = np.asarray(Y_prob)

    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_test, axis=1)

    # plt with given label
    plt.hist(Y_pred_prob[Y_true == label], alpha=0.5, color='red',
             bins=10, log=True, histtype='step')
    plt.hist(Y_pred_prob[Y_true != label], alpha=0.5, color='blue',
             bins=10, log=True, histtype='step')
    # plt.legend(['digit == {}'.format(label), 'digit != {}'.format(label)],
    #            loc='upper right')
    plt.legend([r'class {} matching'.format(label),
               r"class {} not matching".format(label)], loc='best')
    plt.xlabel(r'Probability of being class {}'.format(label))
    plt.ylabel(r'Number of entries')
    plt.savefig("build/{}.pdf".format(fname))
    plt.clf()


def plot_confusion_matrix(cm, classes, fname,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken straight vom SKLEARN.
    """
    print("plot confusion matrix")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("build/{}.pdf".format(fname))
    plt.clf()


def display_errors(errors_index,
                   img_errors,
                   pred_errors,
                   obs_errors,
                   height,
                   width,
                   fname):
    """ This function shows 6 images with their predicted and real labels"""
    print("plot errors")
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((height, width)),
                                cmap=cm.Greys, interpolation='nearest')
            ax[row, col].set_title("Predicted label:{}\nTrue label :{}".format(
                                  pred_errors[error], obs_errors[error]))
            n += 1

    plt.savefig("build/{}.pdf".format(fname))
    plt.clf()


class HistoryEpoch(Callback):
    """Class for calculating loss and metric after each epoch for
    any given dataset, because trainings and validation loss are not comparable
    when using dropout"""
    def __init__(self, data):
        self.data = data

    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.data
        l, a = self.model.evaluate(x, y, verbose=0)
        self.loss.append(l)
        self.acc.append(a)


def plot_history_epoch(train_hist, val_hist, test_hist, fname):
    """plot train, test and val loss and accuracy for objects of class
    History Epoch"""
    print("plot history epoch")
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_hist.loss)
    plt.plot(val_hist.loss)
    plt.plot(test_hist.loss)
    plt.legend(['Training', 'Validation', 'Testing'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(train_hist.acc)
    plt.plot(val_hist.acc)
    plt.plot(test_hist.acc)
    plt.legend(['Training', 'Validation', 'Testing'], loc='lower right')
    plt.savefig("build/{}.pdf".format(fname))
    plt.clf()
