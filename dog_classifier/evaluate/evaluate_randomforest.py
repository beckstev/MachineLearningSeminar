import os
import pickle
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score

from dog_classifier.net.dataloader import DataGenerator
import dog_classifier.autoencoder.randomforest as dog_rf


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


def get_rf_prediction(Dataloader):
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
                                   "autoencoder")
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
    print(diff)
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


def create_confusion_matrix(label_encoder, y_true, y_pred):
    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder")

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = label_encoder.inverse_transform(np.arange(max(y_true)+1))
    if len(classes) == 120:
        mpl.rcParams.update({'font.size': 3})
    else:
        mpl.rcParams.update({'font.size': 5})
    tick_marks = np.arange(max(y_true)+1)

    plt.figure(figsize=(5.8, 3.58))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(tick_marks, classes, rotation=45, horizontalalignment='right')
    plt.yticks(tick_marks, classes)
    plt.title('Confusion matrix random forest\n')
    plt.colorbar()

    # print text if not 120 classes are given
    if max(y_true) != 120:
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
    model_save_path = os.path.join(model_save_path, 'build/confusion_matrix.pdf')
    plt.savefig(model_save_path, dip=600, bbox_inches='tight', pad_inches=0)
    plt.clf()
    # reset rcParams
    mpl.rcParams.update(mpl.rcParamsDefault)


def visualize_rf_preduction(encoder, img_resize):
    ''' Function to visualize the predictions of the random forest.
        :param encoder: File name of the label encoder
        :param img_resize: Tuple (width, height) which defines the size
                           for the rescaled images. The images will be
                           rescaled by the dataloader
    '''

    test_dataloader = get_test_datagenerator(encoder, img_resize)
    y_pred = get_rf_prediction(test_dataloader)
    y_true, img_paths = get_true_labels_and_img_paths(test_dataloader, y_pred.shape[0])
    label_encoder = get_label_encoder(encoder)
    create_confusion_matrix(label_encoder, y_true, y_pred)
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
