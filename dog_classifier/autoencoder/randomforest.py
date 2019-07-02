import os
import json
import pickle
import numpy as np
from pathlib import Path
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier

from dog_classifier.net.train import get_train_and_val_dataloader
from dog_classifier.evaluate.evaluate_training import model_loader
from dog_classifier.evaluate.evaluate_randomforest import get_true_labels_and_img_paths

from dog_classifier.evaluate import evaluate_randomforest as eval_rf



def get_encoder(path_to_autoencoder):
    ''' Function to load a saved autoencoder and convert it into an encoder.
    :param path_to_autoencoder: Path to the directory in which the parameters are
                                saved.
    :return encoder: Return the into a encoder converted autoencoder
    '''
    main_model = 'autoencoder_parameter.h5'
    checkpoint_model = 'autoencoder_parameter_checkpoint.h5'
    autoencoder = model_loader(path_to_autoencoder, main_model, checkpoint_model)
    # Detach the decoder part of the autoencoder
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('flatten_1').output)
    return encoder


def get_labels(Dataloader, len_y_predict):
    ''' Function to return the true labels of a given dataloader. Be sure that you
        used the same Dataloader as you used for training. Otherwise the
        sequence of images may differ.
        :param Dataloader: Dataloader which was used for training/evaluation
        :len_y_predict:: Length of the predicted labels. This value is needed
                         because the batchwise processing of the images by the
                         dataloader can lead to reduction of the total number
                         of images. This happens whenever
                         #images // batch_size != 0
                         then the last batch is not completly filled and
                         will be drpped.
        : return y: Numpy array with the true labels of the images with the
                    same length as the predicted ones.
                    Also the sequence is matching.
    '''
    # Get the dataframe of the dataloader to access the race_labels later
    df = Dataloader.df
    # Dataloader shuffels the labels, therefore we need the mask
    y_mask = Dataloader.data_index
    diff = (y_mask.shape[0] - len_y_predict)

    y = df['race_label'].values
    y = y[y_mask]
    # Y_test erstellen, indem die verwendeten Indizes der Bilder verwendet
    # werden. Dann werden die gedroppt, die überstehen
    if diff is not 0:
        y = y[:-diff]

    return y


def train_random_forest(training_parameters):
    ''' Function to train and save a random forest. The random forest
        will be saved as binary file with pickle.
        :param training_parameters: Dict object which contains all the required training
                                    information/parameters.
    '''
    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder")

    encoder = get_encoder(model_save_path)
    trainDataloader, valDataloader = get_train_and_val_dataloader(training_parameters, is_autoencoder=True)

    X_train_data = encoder.predict_generator(trainDataloader, verbose=1)
    X_val_data = encoder.predict_generator(valDataloader, verbose=1)

    # Because we are not training a NN we do not need a validation  dataset
    # Hence, we can use this data also for training. To check how the random
    # forest can generalize a cross validation is used. However, the cross
    # validation does not use the saved model!
    X_train = np.concatenate((X_train_data, X_val_data))

    y_train_data, _ = get_true_labels_and_img_paths(trainDataloader, X_train_data.shape[0])
    y_val_data, _ = get_true_labels_and_img_paths(valDataloader, X_val_data.shape[0])
    y = np.concatenate((y_train_data, y_val_data))

    # With n_jobs=-1 the random forest always uses the maximal available number
    # of cores
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=10, n_jobs=-1)
    rf.fit(X_train, y)

    pickle_file = 'randomforest.sav'
    rf_save_path = os.path.join(model_save_path, pickle_file)

    with open(rf_save_path, 'wb') as pickle_file:
        pickle.dump(rf, pickle_file)

    # The score does not have to fit the real accuracy of the fitted model
    # due to the fact that the eval_rf_training runs intern a cross validation.
    score = eval_rf.eval_rf_training(rf, X_train, y)

    save_score_path = os.path.join(model_save_path, 'score.json')
    with open(save_score_path, 'w') as json_file:
        json.dump(score, json_file)
