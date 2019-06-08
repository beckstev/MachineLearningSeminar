import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_history(network_history):
    """function for plotting the loss and the accuracy"""
    print("plot history")
    plt.figure()
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend([r'Training', r'Validation'])
    plt.savefig("build/history_loss.pdf")
    plt.clf()

    plt.figure()
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend([r'Training', r'Validation'], loc='lower right')
    plt.savefig("build/history_acc.pdf")
    plt.clf()


def prob_multiclass(Y_pred, Y_test, label):
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
    plt.savefig("build/multiclass.pdf")
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
