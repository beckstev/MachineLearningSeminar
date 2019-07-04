import pandas as pd
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, History, ModelCheckpoint
from keras import backend as K
import json
import sys
import shutil

from dog_classifier.net.dataloader import DataGenerator
from dog_classifier.net.network import DogNN, DogNNv2, DogNNv3, MiniDogNN, PreDogNN, PreBigDogNN
from dog_classifier.evaluate import evaluate_training


def get_model(model_name, n_classes, l2_reg):
    if model_name == 'DogNN':
        return DogNN(n_classes, l2_reg)
    elif model_name == 'DogNNv2':
        return DogNNv2(n_classes, l2_reg)
    elif model_name == 'LinearNN':
        return LinearNN(n_classes, l2_reg)
    elif model_name == 'MiniDogNN':
        return MiniDogNN(n_classes, l2_reg)
    elif model_name == 'DogNNv3':
        return DogNNv3(n_classes, l2_reg)
    elif model_name == 'SeminarNN':
        return SeminarNN(n_classes, l2_reg)
    else:
        raise NameError(f'There is no such Network: {model_name}')


def save_history(history, model_save_path):
    df_history = pd.DataFrame(history.history)
    df_history.to_csv(path_or_buf=model_save_path + '/model_history.csv',
                      index=False)


def save_training_parameters(training_parameters, model_save_path):
    with open(model_save_path + '/training_parameters.json', 'w') as json_file:
        json.dump(training_parameters, json_file)


def save_final_loss_and_acc(history, model_save_path):
    df_history = pd.DataFrame(history.history)
    df_history = df_history.drop(['lr'], axis=1)

    loss_dict = dict()
    # best values
    loss_dict['min_loss'] = min(df_history['loss'].values)
    loss_dict['max_acc'] = max(df_history['acc'].values)
    loss_dict['min_val_loss'] = min(df_history['val_loss'].values)
    loss_dict['max_val_acc'] = max(df_history['val_acc'].values)

    # final values
    loss_dict['loss'] = df_history['loss'].tail(1).values[0]
    loss_dict['acc'] = df_history['acc'].tail(1).values[0]
    loss_dict['val_loss'] = df_history['val_loss'].tail(1).values[0]
    loss_dict['val_acc'] = df_history['val_acc'].tail(1).values[0]

    with open(model_save_path + '/loss_acc.json', 'w') as json_file:
        json.dump(loss_dict, json_file)


def trainNN(training_parameters, grid_search=False):
    ''' Traning a specific net architecture. Afterwards the paramters of the net
        and loss-epoch plot will be saved into saved_models.
    :param training_parameters: Dict which contains all the required traning
                                parameters such as batch size, learning rate.
    :param grid_search: is used to determine if this is for grid search
    :return 0:
    '''

    training_timestamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")
    # Test, if grid_search. in this case, the path has to be modified
    if grid_search:
        model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                       "saved_models",
                                       training_parameters['architecture'],
                                       'hyper_param_tuning',
                                       'bs_'+str(training_parameters['batch_size']) + '_'
                                       'lr_'+str(training_parameters['learning_rate']) + '_'
                                       'l2_'+str(training_parameters['l2_regularisation']))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    else:
        model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                       "saved_models",
                                       training_parameters['architecture'],
                                       training_timestamp)
        os.makedirs(model_save_path)

    n_classes = training_parameters['n_classes']
    encoder_model = training_parameters['encoder_model']
    bs_size = training_parameters['batch_size']
    l2_reg = training_parameters['l2_regularisation']
    num_of_epochs = training_parameters['n_epochs']
    early_stopping_patience = training_parameters['early_stopping_patience']
    early_stopping_delta = training_parameters['early_stopping_delta']

    model = get_model(training_parameters['architecture'], n_classes, l2_reg)
    # Set the leranrning rate of adam optimizer
    Adam(training_parameters['learning_rate'])

    df_train = pd.read_csv(path_to_labels + 'train_labels.csv')
    df_val = pd.read_csv(path_to_labels + 'val_labels.csv')
    with K.tf.device('/cpu:0'):
        trainDataloader = DataGenerator(df_train, encoder_model,
                                        batch_size=bs_size,
                                        n_classes=n_classes)
        valDataloader = DataGenerator(df_val, encoder_model,
                                      batch_size=bs_size, n_classes=n_classes)

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    earlystopper = EarlyStopping(monitor='val_loss',
                                 patience=early_stopping_patience,
                                 min_delta=early_stopping_delta,
                                 verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, verbose=1, min_lr=1e-7)
    # History callback function seperate the model history and the model itself
    # This useful because we can plot/save the history of a model after a
    # KeyboardInterrupt
    hist = History()

    modelCheckpoint = ModelCheckpoint(filepath=model_save_path + '/model_parameter_checkpoint.h5',
                                      verbose=1,
                                      save_best_only=True,
                                      period=2,
                                      save_weights_only=False)

    # We use try to stop the training whenever we want
    try:
        history = model.fit_generator(trainDataloader,
                                      validation_data=valDataloader,
                                      epochs=num_of_epochs,
                                      callbacks=[earlystopper, reduce_lr, hist,
                                                 modelCheckpoint])

    except KeyboardInterrupt:
        print('KeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)')
        save = str(input())
        if save is 'y':
            save_history(hist, model_save_path)
            save_training_parameters(training_parameters, model_save_path)
            evaluate_training.plot_history(hist, path=model_save_path)
        else:
            print(f'Deleting: "{model_save_path}" !')
            shutil.rmtree(model_save_path)
        sys.exit(1)

    model.save(model_save_path + '/model_parameter.h5')
    save_history(history, model_save_path)
    save_training_parameters(training_parameters, model_save_path)
    evaluate_training.plot_history(history, path=model_save_path)

    # if grid_search, additionally save the final (val-)loss and (val-)accuracy
    if grid_search:
        save_final_loss_and_acc(history, model_save_path)
