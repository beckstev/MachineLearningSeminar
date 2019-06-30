from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, Dropout, PReLU, Dropout
from keras.initializers import he_normal
from keras.applications import InceptionResNetV2, MobileNetV2

from keras.regularizers import l2
from keras import backend as K
import argparse
from dog_classifier.net import train

# ------------------------------------------------------------
# needs to be defined as activation class otherwise error
# AttributeError: 'Activation' object has no attribute '__name__'


class PRELU(PReLU):
    def __init__(self, **kwargs):
        self.__name__ = "PReLU"
        super(PRELU, self).__init__(**kwargs)


def DogNN(n_classes, l2_reg):
    # K.set_image_dim_ordering('th')
    shape_input = (None, None, 3)

    model = Sequential()
    # number of params = ( kernel_size * channels + 1) * filters, +1 is bias
    model.add(Conv2D(filters=8, kernel_size=(7, 7), input_shape=shape_input))

    model.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(120, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def DogNNv2(n_classes,l2_reg):
    # K.set_image_dim_ordering('th')
    shape_input = (None, None, 3)
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(3, 3),
                     dilation_rate=(2, 2),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),
                     input_shape=shape_input))
    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None,
                    shared_axes=[1, 2]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=8, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None,
                    shared_axes=[1, 2]))
    model.add(Conv2D(filters=8, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(), weights=None,
                    shared_axes=[1, 2]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(), weights=None,
                    shared_axes=[1, 2]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(), weights=None,
                    shared_axes=[1, 2]))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    return model


def DogNNv3(n_classes, l2_reg):
    # K.set_image_dim_ordering('th')
    shape_input = (None, None, 3)
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(3, 3),
                     dilation_rate=(2, 2),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),
                     input_shape=shape_input))
    model.add(PRELU(alpha_initializer=he_normal(),
                    shared_axes=[1, 2]))
    model.add(Conv2D(filters=8, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(),
                    shared_axes=[1, 2]))
    model.add(Conv2D(filters=25, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(),
                    shared_axes=[1, 2]))
    model.add(Conv2D(filters=40, kernel_size=(3, 3),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),))
    model.add(PRELU(alpha_initializer=he_normal(),
                    shared_axes=[1, 2]))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    return model



def MiniDogNN(n_classes, l2_reg):
    shape_input = (None, None, 3)
    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=(2, 2),
                     dilation_rate=(2, 2),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),
                     input_shape=shape_input))

    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None,
                    shared_axes=[1, 2]))

    model.add(Conv2D(filters=18, kernel_size=(2, 2),
                     dilation_rate=(2, 2),
                     kernel_initializer=he_normal(),
                     bias_initializer=he_normal(),
                     input_shape=shape_input))

    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None,
                    shared_axes=[1, 2]))

    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(20, activation='softmax', kernel_regularizer=l2(l2_reg)))
    model.add(Dense(n_classes, activation='softmax'))

    return model

def PreDogNN():
    l2_value = 0.01
    drop_rate = 0.2

    conv_base = InceptionResNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=(224, 224, 3),
                                  )
    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(30, kernel_initializer=he_normal(),
                    bias_initializer=he_normal(),
                    kernel_regularizer=l2(l2_value)))
    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(30, kernel_initializer=he_normal(),
                    bias_initializer=he_normal(),
                    kernel_regularizer=l2(l2_value)))
    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(5, activation='softmax'))

    return model

def PreBigDogNN(n_classes):
    l2_value = 0.01
    drop_rate = 0.2

    conv_base = InceptionResNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=(224, 224, 3),
                                  )
    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(150, kernel_initializer=he_normal(),
                    bias_initializer=he_normal(),
                    kernel_regularizer=l2(l2_value)))
    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None))
    #model.add(Dropout(rate=drop_rate))
    model.add(Dense(120, kernel_initializer=he_normal(),
                    bias_initializer=he_normal(),
                    kernel_regularizer=l2(l2_value)))
    model.add(PRELU(alpha_initializer=he_normal(),
                    weights=None))

    #model.add(Dropout(rate=drop_rate))
    model.add(Dense(n_classes, activation='softmax'))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: Check number of parameters of a given architecure')
    parser.add_argument('architecture', type=str, help='Class name of the network')
    parser.add_argument('n_classes', type=int, help='number of classes')
   
    l2 = 0.01
    args = parser.parse_args()
    model = train.get_model(args.architecture, args.n_classes, l2)

    model.summary()
