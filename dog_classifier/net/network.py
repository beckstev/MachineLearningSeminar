from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout
from keras import backend as K


def DogNN():
    # K.set_image_dim_ordering('th')
    shape_input = (None, None, 3)

    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(7, 7), activation='relu',
                     input_shape=shape_input))
    model.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(120, activation='softmax'))
    return model


def SeminarNN():
    img_rows, img_cols = None, None

    if K.image_data_format() == 'channels_first':
        shape_input = (3, img_rows, img_cols)
    else:  # channel_last
        shape_input = (img_rows, img_cols, 3)

    # print(shape_input)
    # shape_input = (128, 128, 3)
    # shape_input = (None, None, None, 3)

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='valid',
                     activation='relu', input_shape=shape_input))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                     activation='relu'))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid',
    #                  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())
    # model.add(Flatten())
    model.add(Dense(120, activation='softmax'))

    return model
