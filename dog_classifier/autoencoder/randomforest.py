from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path
import sys
import shutil
from sklearn.externals import joblib
import pickle

from keras.models import Model
from keras.utils import to_categorical

from dog_classifier.evaluate.evaluate_training import model_loader
from dog_classifier.net.train import get_train_and_val_dataloader

def get_encoder(path_to_autoencoder):
    main_model = 'autoencoder_parameter.h5'
    checkpoint_model = 'autoencoder_parameter_checkpoint.h5'
    autoencoder = model_loader(path_to_autoencoder, main_model, checkpoint_model)
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('flatten_1').output)
    return encoder

def get_labels(Dataloader, len_y_predict):
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
    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "auto_encoder")

    encoder = get_encoder('/home/beckstev/Documents/MachineLearningSeminar/saved_models/auto_encoder/28-06-2019_12:18:50')
    trainDataloader, valDataloader = get_train_and_val_dataloader(training_parameters, is_autoencoder=True)

    X_train_data = encoder.predict_generator(trainDataloader, verbose=1)
    X_val_data = encoder.predict_generator(valDataloader, verbose=1)
    X_train = np.concatenate((X_train_data, X_val_data))

    y_train_data = get_labels(trainDataloader, 464)
    y_val_data = get_labels(valDataloader, 192)
    y = np.concatenate((y_train_data, y_val_data))

    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=4)

    rf.fit(X_train, y)

    pickle_file = 'randomforest.sav'
    rf_save_path = os.path.join(model_save_path, pickle_file)

    with open(rf_save_path, 'wb') as pickle_file:
        pickle.dump(rf, pickle_file)
