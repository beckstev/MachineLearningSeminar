from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path
import sys
import shutil

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
    # Y_test erstellen, indem die verwendeten Indizes der Bilder verwendet
    # werden. Dann werden die gedroppt, die überstehen
    y = df['race_label'].values
    y = y[y_mask]
    print(y)
    y = to_categorical(y, num_classes=None)
    # Convert validation observations to one hot vectors
    y = np.argmax(np.array(y), axis=1)
    print(y)
    return y


def train_random_forest(training_parameters):
    encoder = get_encoder('/home/beckstev/Documents/MachineLearningSeminar/saved_models/auto_encoder/28-06-2019_12:18:50')
    trainDataloader, valDataloader = get_train_and_val_dataloader(training_parameters, is_autoencoder=True)

    #X_train_data = encoder.predict_generator(trainDataloader, verbose=1)
    #X_val_data = encoder.predict_generator(valDataloader, verbose=1)
    #print(X_train_data.shape, X_val_data.shape)

    #X_train = np.concatenate((X_train_data, X_val_data))

    y_train_data = get_labels(trainDataloader, 464)
    y_val_data = get_labels(valDataloader, 192)

    y = np.concatenate((y_train_data, y_val_data))
    print(y)
