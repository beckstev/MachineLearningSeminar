import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import os
from keras.callbacks import Callback
from keras.layers import PReLU
from sklearn.metrics import classification_report
from keras.models import load_model
from dog_classifier.net.dataloader import DataGenerator
from keras.utils import to_categorical
import keras.backend as K
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
from keras.utils.generic_utils import CustomObjectScope
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from dog_classifier.evaluate.tab2tex import make_table

mpl.use('pgf')
mpl.rcParams.update(
    {'font.size': 10,
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'lualatex',
        'text.latex.unicode': True,
        'pgf.preamble': r'\DeclareMathSymbol{.}{\mathord}{letters}{"3B}',
     })


class HistoryEpoch(Callback):
    """Class for calculating loss and metric after each epoch for
    any given dataset, because trainings and validation loss are not comparable
    when using dropout. Callback can be used to get internal states and
    statistics during training, in our case after each epoch.

    :param Callback: object of type Callback
    """
    def __init__(self,  datagenerator):
        self.datagenerator = datagenerator

    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.lr = []

    def on_epoch_end(self, epoch, logs={}):
        l, a = self.model.evaluate_generator(self.datagenerator, verbose=0)
        lr = K.eval(self.model.optimizer.lr)
        self.loss.append(l)
        self.acc.append(a)
        self.lr.append(lr)


def plot_history(network_history, path):
    """function for plotting the loss and the accuracy

    :param network_history: result of fitting the model
    :param path: saving path
    """
    print("plot history")
    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)

    num_epochs = len(network_history.history['loss'])
    epoche = np.arange(1, num_epochs + 1)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 3.58), sharex=True)

    ax0.set_ylabel('Loss')
    ax0.plot(epoche, network_history.history['loss'], label='Training')
    ax0.plot(epoche, network_history.history['val_loss'], label='Validation')

    ymin, ymax = ax0.get_ylim()
    y_ticks = np.round(np.linspace(ymin, ymax, 4), 2)
    ax0.set_yticks(y_ticks)
    ax0.legend()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epoche, network_history.history['acc'])
    ax1.plot(epoche, network_history.history['val_acc'])

    ymin, ymax = ax1.get_ylim()
    y_ticks = np.round(np.linspace(ymin, ymax, 4), 2)
    ax1.set_yticks(y_ticks)
    plt.tight_layout()
    plt.savefig("{}/history.pdf".format(path), dpi=500,
                pad_inches=0, bbox_inches='tight')
    plt.clf()


def prob_multiclass(Y_pred, Y_test, Y_true, label, path):
    """defines a multiclass probability in the case that the output of the cnn
    cannot be interpreted as a probability. Also plots for a given label

    :param Y_pred: model prediction of the test data
    :param Y_test: test output data
    :param label: number (or rather name?) of class, which should be examined
    :param path: saving path
    """
    print("plot multiclass probability")
    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)
    n_cls = len(Y_pred[0])

    Y_prob = []
    for i in range(len(Y_pred)):
        numerator = Y_pred[i, label]
        denominator = 0.0
        for idx in range(n_cls):
            denominator += Y_pred[i, idx]

        Y_prob.append(numerator/denominator)

    Y_pred_prob = np.asarray(Y_prob)

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
    plt.savefig("{}/multiclass.pdf".format(path))
    plt.clf()


def plot_confusion_matrix(cm, classes, path, encoder_model,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken straight vom SKLEARN.

    :param cm: confusion matrix generated by sklearn
    :param classes: range with lenght of classes
    :param encoder_model: encoder_model
    :param path: saving path
    """
    # Set figsize
    plt.figure(figsize=(5.8, 3.58))
    # change font size according to number of classes
    n_classes = len(classes)
    print(n_classes)


    print("plot confusion matrix")

    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Decode the class names
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    encoder_path = os.path.join(path_to_labels, encoder_model)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_path)
    classes = encoder.inverse_transform(classes)
    classes = [cl.replace('_', ' ') for cl in classes]

    if n_classes == 120:
        mpl.rcParams.update({'font.size': 3})
    else:
        mpl.rcParams.update({'font.size': 5})

    # Check if normalize is True, then scale the colorbar accordingly
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if n_classes == 120:
        plt.title(title, fontsize=10)
    else:
        plt.title(title)

    plt.colorbar()
    tick_marks = np.arange(n_classes)
    if n_classes == 120:
        ticks = range(1, n_classes+1)
        plt.axes().xaxis.set_major_locator(FixedLocator(ticks[0::2]))
        plt.axes().xaxis.set_minor_locator(FixedLocator(ticks[1::2]))
        plt.axes().xaxis.set_minor_formatter(FormatStrFormatter("%d"))
        plt.axes().tick_params(which='major', pad=6, axis='x', rotation=90, labelsize=2)
        plt.axes().tick_params(which='minor', pad=0.5, axis='x', rotation=90, labelsize=4)

        plt.axes().yaxis.set_major_locator(FixedLocator(ticks[0::2]))
        plt.axes().yaxis.set_minor_locator(FixedLocator(ticks[1::2]))
        plt.axes().yaxis.set_minor_formatter(FormatStrFormatter("%d"))
        plt.axes().tick_params(which='major', pad=6, axis='y', labelsize=2)
        plt.axes().tick_params(which='minor', pad=0.5, axis='y', labelsize=4)
        #plt.xticks(horizontalalignment='right')

    else:
        plt.xticks(tick_marks, classes, rotation=45, horizontalalignment='right')
        plt.yticks(tick_marks, classes)

    # print text if not 120 classes are given
    if n_classes != 120:
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
    plt.savefig("{}/confusion_matrix.pdf".format(path), dpi=500, pad_inches=0, bbox_inches='tight')
    plt.clf()
    # reset rcParams
    mpl.rcParams.update(mpl.rcParamsDefault)
    if n_classes == 120:
        header = ['Label', 'Hunderasse']
        places = [1.0, 1.0]
        data = [ticks, classes]
        caption = 'Legende: Label - Hunderasse'
        label = 'tab:legende_rf'
        filename = os.path.join(path, 'legende.tex')
        make_table(header, places, data, caption, label, filename)


def display_errors(n, Y_cls, Y_true, Y_pred, X_test, height, width, nrows,
                   ncols, path):
    """ This function computes the n-th biggest errors and shows n
    images with their predicted and real labels.

    :param n: number of plots
    :param Y_cls: predicition classes as hot vector
    :param Y_true: validation observation as hot vector
    :param X_test: test input dataset
    :param height: height of single plot
    :param width: width of single plot
    :param nrows: number of rows in subplot
    :param ncols: number of columns in subplot
    :param path: saving path
    """
    print("plot errors")
    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)
    errors = (Y_cls - Y_true != 0)

    Y_cls_errors = Y_cls[errors]
    Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    X_test_errors = X_test[errors]

    # rank errors in probability
    # Probabilities of the wrong predicted numbers
    Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors,
                                           axis=1))

    # Difference between the probability of the predicted label and the true
    # label
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)

    # Top n errors
    most_important_errors = sorted_dela_errors[-n:]

    # just renaming for clarity
    errors_index = most_important_errors
    img_errors = X_test_errors
    pred_errors = Y_cls_errors
    obs_errors = Y_true_errors

    n = 0
    nrows = nrows
    ncols = ncols
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((height, width)),
                                cmap=cm.Greys, interpolation='nearest')
            ax[row, col].set_title("Predicted label:{}\nTrue label :{}".format(
                                  pred_errors[error], obs_errors[error]))
            n += 1

    plt.savefig("{}/errors.pdf".format(path))
    plt.clf()


def plot_history_epoch(train_hist, val_hist, path):
    """plot train, test and val loss and accuracy for objects of class
    History Epoch

    :param train_hist: Object of class History Epoch given of train input
                        and output data
    :param val_hist: Object of class History Epoch given of validation input
                        and output data

    :param path: name of resulting plot
    """
    print("plot history epoch")
    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)

    # First x-axis entry should start at 1
    num_epochs = len((train_hist.loss))
    epoche = np.arange(1, num_epochs + 1)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 3.58), sharex=True)
    ax0.set_ylabel('Loss')
    ax0.plot(epoche, train_hist.loss, label='Training')
    ax0.plot(epoche, val_hist.loss, label='Validation')

    ymin, ymax = ax0.get_ylim()
    y_ticks = np.round(np.linspace(ymin, ymax, 4), 2)
    ax0.set_yticks(y_ticks)
    ax0.legend()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epoche, train_hist.acc)
    ax1.plot(epoche, val_hist.acc)

    ymin, ymax = ax1.get_ylim()
    y_ticks = np.round(np.linspace(ymin, ymax, 4), 2)
    ax1.set_yticks(y_ticks)
    plt.tight_layout()
    plt.savefig("{}/history_epoch.pdf".format(path), dpi=500,
                pad_inches=0, bbox_inches='tight')
    plt.clf()


def evaluate_test_params(model, X_test, Y_test):
    """evaluates the loss and the accuracy of the test dataset and prints
    the classification report

    :param model: active NN
    :param X_test: test input dataset
    :param Y_test: test output dataset
    """
    print("evaluate test params")
    # Evaluate loss and metrics
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)
    # Predict the values from the test dataset
    Y_pred = model.predict(X_test)
    # Convert predictions classes to one hot vectors
    Y_cls = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_test, axis=1)
    print('Classification Report:\n', classification_report(Y_true, Y_cls))


def plot_predictions(n, model, X_test, height, width, path):
    """plot the first n predicitions.

    :param n: number of predicions
    :param model: active NN
    :param X_test: test input dataset
    :param height: height of given pictures
    :param width: width of given pictures
    :param path: name of resulting plot
    """
    print("plot predictions")
    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)
    slice = n
    predicted = model.predict(X_test[:slice]).argmax(-1)
    # plt.figure(figsize=(5.8, 3.58))
    for i in range(slice):
        plt.subplot(1, slice, i+1)
        plt.imshow(X_test[i].reshape(height, width), interpolation='nearest')
        plt.text(0, 0, predicted[i], color='black',
                 bbox=dict(facecolor='white', alpha=1))
        plt.axis('off')
    plt.savefig("{}/plot_predictions.pdf".format(path))


def predict(path_to_model, encoder_model, fname, img_resize, use_rgb):
    """function will predict with predict_generator from a given, saved model
    and save the result as txt
    :param path_to_model: Path to model
    :param encoder_model: version of encoder used for training
    :param fname: name of the created predicition txt file
    :param img_resize: Tuple (width, height) with determines the shape of the resized images
    """
    main_model = 'model_parameter.h5'
    checkpoint_model = 'model_parameter_checkpoint.h5'
    model_params = os.path.join(path_to_model, main_model)
    # This try and except block enables us to use 'model_parameter_checkpoint.h5'
    # files if there is now 'model_parameter_checkpoint.h5'

    with CustomObjectScope({'PRELU': PReLU()}):
        model = model_loader(path_to_model, main_model, checkpoint_model)

    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")
    df_test = pd.read_csv(path_to_labels + 'test_labels.csv')

    testDataloader = DataGenerator(df_test,
                                   encoder_model=encoder_model,
                                   shuffle=True,
                                   is_test=True,
                                   const_img_resize=img_resize,
                                   use_rgb=use_rgb)

    # Test predicten
    Y_pred = model.predict_generator(testDataloader, verbose=1)

    # Save predictions
    path_predictions = os.path.join(path_to_model, fname)
    np.savetxt(path_predictions, Y_pred)


def preprocess(path_to_model, encoder_model, fname):
    """function, that computes Y_pred, Y_test, Y_cls and Y_true for further use
    in the evaluation process.
    :param path_to_model: path to model, where everthing regarding this model
                          is saved
    :param encoder_model: version of encoder used for training
    :param fname: name of predicition txt
    """

    # Read test dataset
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")
    df_test = pd.read_csv(path_to_labels + 'test_labels.csv')

    # call the data generator for the test loader
    testDataloader = DataGenerator(df_test,
                                   encoder_model=encoder_model,
                                   shuffle=True,
                                   is_test=True,
                                   )

    # create prediction array from given file
    path_predictions = os.path.join(path_to_model, fname)

    Y_pred = np.genfromtxt(path_predictions)
    # The Dataloader converts the labels automatically into ahot vecotor
    Y_test = testDataloader.df['race_label'].values
    diff = (Y_test.shape[0] - Y_pred.shape[0])

    # Y_test erstellen, indem die verwendeten Indizes der Bilder verwendet
    # werden. Dann werden die gedroppt, die überstehen
    Y_true = Y_test[testDataloader.data_index]
    # if diff is equal to zero we get an empty array. We dont want this.
    if diff is not 0:
        Y_true = Y_true[:-diff]

    # Convert predictions classes to one hot vectors
    Y_cls = np.argmax(np.array(Y_pred), axis=1)

    # Convert validation observations to one hot vectors
    path_to_images = df_test['path_to_image'].values
    path_to_images = path_to_images[testDataloader.data_index]
    return Y_pred, Y_test, Y_cls, Y_true, path_to_images


def visualize_predictions(Y_pred, Y_true, path_to_images, encoder_model, path):
    print('\nvisualize predictions \n')
    plt.figure(figsize=(6.224, 4))

    path = path + '/build/'
    if not os.path.exists(path):
        os.makedirs(path)

    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    encoder_path = os.path.join(path_to_labels, encoder_model)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_path)

    for index in range(24, 30):
        # Indices of the array represent the dog race
        race_index_true = Y_true[index]
        race_true = encoder.inverse_transform(np.array([race_index_true]))
        race_true = [cl.replace('_', ' ') for cl in race_true]

        # Indicies of the three races with highest prohability
        # Notice: the last element of races_high_pred has the highest prob
        races_high_pred = np.argsort(Y_pred[index])[-1:][::-1]
        races_pred = encoder.inverse_transform(races_high_pred)

        plt.subplot(2, 3, index-23)

        path_to_image = path_to_images[index]
        img = plt.imread(path_to_image)
        plt.imshow(img)
        img_width = img.shape[1]
        img_height = img.shape[0]
        # scale = 6
        # height_steps = img_height / 6

        for i in range(len(races_pred)):
            race_index_pred = races_high_pred[i]
            prob = Y_pred[index][race_index_pred] * 100
            text = f"{races_pred[i].replace('_', ' ')}: {prob:.2f} \%"
            # plt.text(img_width * 51/50, (i + scale/2 - 1) * height_steps, text)
            # plt.text(img_width * 0.08, img_height * 1.09, text, fontsize=7)
            plt.annotate(text, (0, 0), (0, -1), xycoords='axes fraction',
                         textcoords='offset points', va='top', fontsize=7)

        plt.title(f'True race: {race_true[0]} ', fontsize=7)
        plt.axis('off')
        # plt.savefig('test.png', bbox_inches='tight', pad_inches=0.05)
        # plt.show()
    plt.savefig("{}/visualize_predictions.pdf".format(path), dpi=500,
                pad_inches=0, bbox_inches='tight')


def model_loader(path_to_model, main_model, checkpoint_model):
    try:
        model_params = os.path.join(path_to_model, main_model)
        model = load_model(model_params)

    except OSError as e:
        files_in_mode_path = os.listdir(path_to_model)

        if checkpoint_model in files_in_mode_path:
            print('-------------')
            print(f'The request model does not include "{main_model}".')
            print(f'However, there is a checkpoint model "{checkpoint_model}"',
                  'which could be used instead.')
            print('Do you wanna evaluate the checkpoint_model? yes-(y), no-(n)')
            reply = str(input())
            if reply == 'y':
                print('Using checkpoint model')
                checkpoint_model_params = os.path.join(path_to_model,
                                                       checkpoint_model)
                model = load_model(checkpoint_model_params)
            else:
                print('Exiting program')
                sys.exit(1)
        else:
            raise OSError(f'There is no file "{main_model}" or "{checkpoint_model}"!')

    return model


if __name__ == '__main__':
    enocder = "encoder_2019-06-16_12:45:37.npy"
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")
    df_test = pd.read_csv(path_to_labels + 'val_test.csv')
