import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, History, ModelCheckpoint
from keras import backend as K
import json
import sys
import shutil

from dog_classifier.net.dataloader import DataGenerator
from dog_classifier.net.network import DogNN, DogNNv2, LinearNN, DogNNv3, MiniDogNN
from dog_classifier.evaluate import evaluate_training


def get_model(model_name):
    if model_name == 'DogNN':
        return DogNN()
    elif model_name == 'DogNNv2':
        return DogNNv2()
    elif model_name == 'LinearNN':
        return LinearNN()
    elif model_name == 'MiniDogNN':
        return MiniDogNN()
    elif model_name == 'DogNNv3':
        return DogNNv3()
    else:
        raise NameError(f'There is no such Network: {model_name}')

def save_history(history, model_save_path):
    df_history = pd.DataFrame(history.history)
    df_history.to_csv(path_or_buf=model_save_path + '/model_history.csv',
                     index=False)
def save_training_parameters(training_parameters, model_save_path):
    with open(model_save_path + '/training_parameters.json', 'w') as json_file:
        json.dump(training_parameters, json_file)

def trainNN(training_parameters):
    ''' Traning a specific net architecture. Afterwards the paramters of the net
        and loss-epoch plot will be saved into saved_models.
    :param training_parameters: Dict which contains all the required traning
                                parameters such as batch size, learning rate.
    :return 0:
    '''

    training_timestamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                       "labels/")
    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   training_parameters['architecture'],
                                   training_timestamp)
    os.makedirs(model_save_path)

    n_classes = training_parameters['n_classes']
    encoder_model = training_parameters['encoder_model']
    bs_size = training_parameters['batch_size']
    num_of_epochs = training_parameters['n_epochs']
    early_stopping_patience = training_parameters['early_stopping_patience']
    early_stopping_delta = training_parameters['early_stopping_delta']

    model = get_model(training_parameters['architecture'])
    # Set the leranrning rate of adam optimizer
    Adam(training_parameters['learning_rate'])

    df_train = pd.read_csv(path_to_labels + 'train_labels.csv')
    df_val = pd.read_csv(path_to_labels + 'val_labels.csv')
    with K.tf.device('/cpu:0'):
        trainDataloader = DataGenerator(df_train, encoder_model,
                                        batch_size=bs_size)
        valDataloader = DataGenerator(df_val, encoder_model,
                                      batch_size=bs_size)

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
        history = model.fit_generator(trainDataloader, validation_data=valDataloader,
                                      epochs=num_of_epochs,
                                      callbacks=[earlystopper, reduce_lr, hist, modelCheckpoint])

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
