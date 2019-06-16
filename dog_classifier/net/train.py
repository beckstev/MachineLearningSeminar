import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import json

from dog_classifier.net.dataloader import DataGenerator
from dog_classifier.net.network import DogNN, DogNNv2, LinearNN
from dog_classifier.evaluate import evaluate_training


def get_model(model_name):
    if model_name == 'DogNN':
        return DogNN()
    elif model_name == 'DogNNv2':
        return DogNNv2()
    elif model_name == 'LinearNN':
        return LinearNN()
    else:
        raise NameError(f'There is no such Network: {model_name}')


def trainNN(training_parameters):
    ''' Traning a specific net architecture. Afterwards the paramters of the net
        and loss-epoch plot will be saved into saved_models.
    :param training_parameters: Dict which contains all the required traning
                                parameters such as batch size, learning rate.
    :return 0:
    '''

    training_timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2], "labels/")

    encoder_model = training_parameters['encoder_model']
    bs_size = training_parameters['batch_size']
    num_of_epochs = training_parameters['n_epochs']
    early_stopping_patience = training_parameters['early_stopping_patience']
    early_stopping_delta = training_parameters['early_stopping_delta']

    df_train = pd.read_csv(path_to_labels + 'train_labels.csv')
    df_val = pd.read_csv(path_to_labels + 'val_labels.csv')

    trainDataloader = DataGenerator(df_train, encoder_model, batch_size=bs_size)
    valDataloader = DataGenerator(df_val, encoder_model, batch_size=bs_size)

    model = get_model(training_parameters['architecture'])
    # Set the leranrning rate of adam optimizer
    Adam(training_parameters['learning_rate'])

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    earlystopper = EarlyStopping(monitor='val_loss',
                                 patience=early_stopping_patience,
                                 min_delta=early_stopping_delta,
                                 verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5)

    history = model.fit_generator(trainDataloader, validation_data=valDataloader,
                                  epochs=num_of_epochs)
                                  #callbacks=[earlystopper, reduce_lr])

    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   training_parameters['architecture'],
                                   training_timestamp)

    os.makedirs(model_save_path)
    model.save(model_save_path + '/model_parameter.h5')
    evaluate_training.plot_history(history, path=model_save_path)
    df_history = pd.Dataframe(history.history)
    df_history.to_csv(path_or_buf=model_save_path + '/model_history.csv',
                      index=False)