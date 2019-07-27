import os
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score

from dog_classifier.evaluate.tab2tex import make_table
from dog_classifier.net.dataloader import DataGenerator
import dog_classifier.autoencoder.randomforest as dog_rf

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


def eval_rf_training(rf, X, y):
    ''' Function to evaluate a random forest with a cross validation.
        Currently three cross validation/splits are used
        :param rf: Random forest model
        :param X: Input data which the random forest has to classify
        :param y: True labels of the input data
        :return score_acc: Dict which contains the mean and std of the
                           accuracy
    '''
    num_cv = 3
    score_cv_acc = cross_validate(rf, X, y,
                                  scoring=make_scorer(accuracy_score),
                                  cv=num_cv)
    score_acc = dict()
    score_acc['mean'] = score_cv_acc['test_score'].mean()
    score_acc['std'] = score_cv_acc['test_score'].std(ddof=1)
    print(f'Accuracy {score_acc["mean"]:0.3f} +/- {score_acc["std"]:0.3f}')

    return score_acc


def get_rf_prediction(Dataloader, n_classes):
    ''' Function to get the prediction of the saved random forest (rf).
        In the moment, the rf model has to be saved in the autoencoder
        directory located in saved_models. In addtion, the model has to
        be saved as binary .sav file with the name "randomforest.sav".
        To generate the correct input for the rf the pretrained encoder
        (AutoDogEncoder) is used.
        :param Dataloader: Dataloader which is required for the prediction
                           of the encoder (AutoDogEncoder)
        :return y_pred: Returns the predicted labels as hot vector
    '''

    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder" + "_n_" + str(n_classes))
    encoder = dog_rf.get_encoder(model_save_path)

    rf_name = 'randomforest.sav'
    path_to_rf = os.path.join(model_save_path, rf_name)
    with open(path_to_rf, 'rb') as f:
        rf = pickle.load(f)

    # Using the encoder to get features for the random forest
    X_rf = encoder.predict_generator(Dataloader, verbose=1)

    #The pretrained rf uses the features to make a label prediction
    y_pred = rf.predict(X_rf)

    return y_pred


def get_test_datagenerator(encoder_model, img_resize):
    ''' Function to get the Dataloader for the test dataset
        :param encoder_model: File name of the label encoder
        :img_reisze: Size of the input images.
        :return testDataloader: Returns dataloader object
    '''

    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    df_test = pd.read_csv(path_to_labels + 'test_labels.csv')

    testDataloader = DataGenerator(df_test,
                                   encoder_model=encoder_model,
                                   shuffle=True,
                                   is_test=True,
                                   use_rgb=True,
                                   const_img_resize=img_resize)

    return testDataloader


def get_true_labels_and_img_paths(dataloader, len_y_predict):
    ''' Function to return the true labels of a given dataloader. Be sure that you
        used the same Dataloader as you used for training. Otherwise the
        sequence of images may differ.
        :param dataloader: Dataloader which was used for training/evaluation
        :len_y_predict:: Length of the predicted labels. This value is needed
                         because the batchwise processing of the images by the
                         dataloader can lead to reduction of the total number
                         of images. This happens whenever
                         #images // batch_size != 0
                         then the last batch is not completly filled and
                         will be drpped.
        :return y_true: Numpy array with the true labels of the images with the
                         same length as the predicted ones.
                         Also the sequence is matching.
        :return path_to_image: Returns a numpy array with the
                               path_to_image with the same sequence as
                               y_pred
    '''
    y = dataloader.df['race_label'].values
    # ataloader.data_index is required because dataloader shuffels the
    # labels hence we need to mask the array
    y_true = y[dataloader.data_index]
    path_to_images = dataloader.df['path_to_image'].values
    path_to_images = path_to_images[dataloader.data_index]

    # Check if there was a uncompleted batch
    diff = (y_true.shape[0] - len_y_predict)
    if diff is not 0:
        y_true = y_true[:-diff]
        path_to_images = path_to_images[:-diff]
    return y_true, path_to_images


def get_label_encoder(encoder_model):
    ''' Function to load and return a preconfigurated label encoder.
        :param encoder_model: File name of the label encoder
        :return encoder: Returns the preconfigurated label encoder
    '''
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    encoder_path = os.path.join(path_to_labels, encoder_model)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_path)
    return encoder


def create_confusion_matrix(label_encoder, n_classes, y_true, y_pred):
    # I have to put this here, othwise I get a strange error:
    # File "/home/beckstev/.local/anaconda3/lib/python3.7/site-packages/matplotlib/dviread.py", line 199, in __init__
    # self.file = open(filename, 'rb')
    # FileNotFoundError: [Errno 2] No such file or directory:
    # '/home/user/.cache/matplotlib/tex.cache/4ee3b5cbb5a9a35df6cd457d83904876.dvi'
    mpl.rcParams.update(mpl.rcParamsDefault)

    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder" + "_n_" + str(n_classes))

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = label_encoder.inverse_transform(np.arange(max(y_true)+1))


    if n_classes == 120:
        mpl.rcParams.update({'font.size': 3})
    else:
        classes = [cl.replace('_', ' ') for cl in classes]
        mpl.rcParams.update({'font.size': 5})
    tick_marks = np.arange(max(y_true)+1)

    plt.figure(figsize=(5.8, 3.58))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)


    if n_classes == 120:
        ticks = range(1, n_classes+1)
        plt.axes().xaxis.set_major_locator(FixedLocator(ticks[0::2]))
        plt.axes().xaxis.set_minor_locator(FixedLocator(ticks[1::2]))
        plt.axes().xaxis.set_minor_formatter(FormatStrFormatter("%d"))
        plt.axes().tick_params(which='major', pad=6, axis='x', rotation=90)
        plt.axes().tick_params(which='minor', pad=0.5, axis='x', rotation=90)

        plt.axes().yaxis.set_major_locator(FixedLocator(ticks[0::2]))
        plt.axes().yaxis.set_minor_locator(FixedLocator(ticks[1::2]))
        plt.axes().yaxis.set_minor_formatter(FormatStrFormatter("%d"))
        plt.axes().tick_params(which='major', pad=6, axis='y')
        plt.axes().tick_params(which='minor', pad=0.5, axis='y')
        #plt.xticks(horizontalalignment='right')

    else:
        plt.xticks(tick_marks, classes, rotation=45, horizontalalignment='right')
        plt.yticks(tick_marks, classes)


    plt.title('Confusion matrix random forest\n')
    plt.colorbar()

    # print text if not 120 classes are given
    if n_classes != 120:
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    model_save_path_matrix = os.path.join(model_save_path, 'build/confusion_matrix.pdf')
    plt.savefig(model_save_path_matrix, dpi=500, pad_inches=0, bbox_inches='tight')
    plt.clf()
    # reset rcParams
    mpl.rcParams.update(mpl.rcParamsDefault)


    if n_classes==120:
        header = ['$\map{Label}$', '$\map{Hunderasse}$']
        places = [1.0, 1.0]
        data = [ticks, classes]
        caption = 'Legende: Label - Hunderasse'
        label = 'tab:legende_rf'
        filename = os.path.join(model_save_path, 'build/legende.tex')
        make_table(header, places, data, caption, label, filename)

def visualize_rf_preduction(encoder, img_resize, n_classes):
    ''' Function to visualize the predictions of the random forest.
        :param encoder: File name of the label encoder
        :param img_resize: Tuple (width, height) which defines the size
                           for the rescaled images. The images will be
                           rescaled by the dataloader
    '''

    test_dataloader = get_test_datagenerator(encoder, img_resize)
    y_pred = get_rf_prediction(test_dataloader, n_classes)
    y_true, img_paths = get_true_labels_and_img_paths(test_dataloader, y_pred.shape[0])
    label_encoder = get_label_encoder(encoder)
    create_confusion_matrix(label_encoder, n_classes, y_true, y_pred)
    for index in range(len(y_pred)):
        race_true = label_encoder.inverse_transform((y_true[index],))[0]
        race_pred = label_encoder.inverse_transform((y_pred[index],))[0]
        img_path = img_paths[index]

        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f'True: {race_true}   -   Prediction: {race_pred}')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    encoder = 'encoder_2019-06-27_19:48:46.npy'
    visualize_rf_preduction(encoder)
