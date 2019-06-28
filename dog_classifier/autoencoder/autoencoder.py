import os
from datetime import datetime
from pathlib import Path
import sys
import shutil

from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, History, ModelCheckpoint

from dog_classifier.net.train import get_train_and_val_dataloader, save_history, save_training_parameters
from dog_classifier.evaluate import evaluate_training


def AutoDogEncoder(img_input_size, n_classes):
    img_input_size = (img_input_size[0], img_input_size[1], 3)
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

    # Output shape:  (None, 15, 15, 64)
    model.add(Flatten())
    model.add(Reshape((30, 30, 16)))
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

    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder")
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    num_of_epochs = training_parameters['n_epochs']

    trainDataloader, valDataloader = get_train_and_val_dataloader(training_parameters, is_autoencoder=True)

    n_classes = training_parameters['n_classes']
    img_resize = training_parameters['img_resize']
    model = AutoDogEncoder(img_resize, n_classes)
    # Set the leranrning rate of adam optimizer
    Adam(training_parameters['learning_rate'])

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, verbose=1, min_lr=1e-7)
    # History callback function seperate the model history and the model itself
    # This useful because we can plot/save the history of a model after a
    # KeyboardInterrupt
    hist = History()

    modelCheckpoint = ModelCheckpoint(filepath=model_save_path + '/autoencoder_parameter_checkpoint.h5',
                                      verbose=1,
                                      save_best_only=True,
                                      period=2,
                                      save_weights_only=False)

    try:
        history = model.fit_generator(trainDataloader, validation_data=valDataloader,
                                      epochs=num_of_epochs,
                                      callbacks=[reduce_lr, hist, modelCheckpoint])

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

    model.save(model_save_path + '/autoencoder_parameter.h5')
    save_history(history, model_save_path)
    save_training_parameters(training_parameters, model_save_path)
    evaluate_training.plot_history(history, path, model_save_path)
