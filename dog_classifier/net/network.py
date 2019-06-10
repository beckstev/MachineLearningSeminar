from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten,GlobalAveragePooling2D, Dropout
from keras import backend as K


def DogNN():
    # K.set_image_dim_ordering('th')
    # shape_input = ( 3, 500, 500)
    shape_input = (500, 500, 3)

    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(7, 7), activation='relu',
                     input_shape=shape_input))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(120, activation='softmax'))

    return model
