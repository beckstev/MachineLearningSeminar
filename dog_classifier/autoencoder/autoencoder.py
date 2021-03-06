import os
import sys
from pathlib import Path

from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, History, ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D

from dog_classifier.evaluate import evaluate_training
from dog_classifier.net.train import get_train_and_val_dataloader, save_history, save_training_parameters


def AutoDogEncoder(img_input_size, n_classes):
    ''' Convolutional autoencoder to generate features for a random forest
        based classifier.
        :param img_input_size: Size of the input images. Set by argparse in
                               train_autoencoder
        :param n_classes: Number of classes we wanna to classify. The idea is to
                          adapt the size of the autoencoder to the number of
                          classes. NOT USED IN THE CURRENT VERSION!
        :return model: Returns the autoencoder as keras Sequential
    '''
    img_input_size = (img_input_size[1], img_input_size[0], 3)
    model = Sequential()
    # Encoder
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                     input_shape=img_input_size, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(filters=8, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(filters=8, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     activation='relu', padding='same'))

    # Output shape is defined by the maxpooling2D
    # [1::] to skip the None value
    final_shape = model.get_layer('conv2d_4').output_shape[1::]
    model.add(Flatten())

    model.add(Reshape(final_shape))
    # Decoder
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(filters=8, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(filters=8, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(Conv2D(filters=3, kernel_size=(5, 5),
                     activation='sigmoid', padding='same'))
    return model


def train_autoencoder(training_parameters):
    ''' Function to train the AutoDogEncoder Autoencoder.
    :param training_parameters: Dict object which contains all the required training
                                information/parameters.

    '''
    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder" + "_n_" + str(training_parameters['n_classes']))

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    num_of_epochs = training_parameters['n_epochs']

    trainDataloader, valDataloader = get_train_and_val_dataloader(training_parameters,
                                                                  is_autoencoder=True)

    n_classes = training_parameters['n_classes']
    img_resize = training_parameters['img_resize']
    model = AutoDogEncoder(img_resize, n_classes)

    # Set the leranrning rate of adam optimizer
    adam = Adam(training_parameters['learning_rate'])

    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, verbose=1, min_lr=1e-7)
    # History callback function seperate the model history and the model itself
    # This useful because we can plot/save the history of a model after a
    # KeyboardInterrupt
    hist = History()

    early_stopping_patience = training_parameters['early_stopping_patience']
    early_stopping_delta = training_parameters['early_stopping_delta']
    earlystopper = EarlyStopping(monitor='val_loss',
                                 patience=early_stopping_patience,
                                 min_delta=early_stopping_delta,
                                 verbose=1)

    modelCheckpoint = ModelCheckpoint(filepath=model_save_path + '/autoencoder_parameter_checkpoint.h5',
                                      verbose=1,
                                      save_best_only=True,
                                      period=2,
                                      save_weights_only=False)

    try:
        history = model.fit_generator(trainDataloader,
                                      validation_data=valDataloader,
                                      epochs=num_of_epochs,
                                      callbacks=[reduce_lr, hist, modelCheckpoint, earlystopper])

    except KeyboardInterrupt:
        print('KeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)')
        save = str(input())
        if save is 'y':
            save_history(hist, model_save_path)
            save_training_parameters(training_parameters, model_save_path)
            evaluate_training.plot_history(hist, path=model_save_path)
        sys.exit(1)

    model.save(model_save_path + '/autoencoder_parameter.h5')
    save_history(history, model_save_path)
    save_training_parameters(training_parameters, model_save_path)
    evaluate_training.plot_history(history, path=model_save_path)
