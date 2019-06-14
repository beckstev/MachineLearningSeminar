import numpy as np
import cv2 as cv
from keras.utils import to_categorical, Sequence


class DataGenerator(Sequence):
    ''' Dataloader for batch wise traning of a NN with batch wise rescaling of
        images. The function inherits the properties of keras.utils.Sequence
        so that we can leverage nice functionalities such as multiprocessing.
    '''

    def __init__(self, df, batch_size=16, n_classes=120,
                 use_rgb=True, shuffle=True, seed=13):
        '''Initialization
        :param df: Dataframe of the dataset (training, validation or testing)
                   which contains for all images the path to the image, the
                   label and the width & height of the image
        :param batch_size:  Size of the batch
        :param n_classes: Number of classes you wanna classify
        :param use_rgb: Boolean to indicate if the function should use RGB
                        or grayscale images
        :param shuffle: To get a new image order every epoch abs
        :param seed: Set a seed for numpy random functions
        '''
        self.df = df
        self.batch_size = batch_size
        self.number_IDs = len(df)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_rgb = use_rgb
        self.seed = seed
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.data_index = np.arange(self.number_IDs)

        if self.shuffle is True:
            np.random.seed(self.seed)
            np.random.shuffle(self.data_index)

    def __data_generation(self, list_IDs_temp):
        ''' Function which load and rescaled all the images for batch. The
            images will be scaled to the size of the smallest image in a batch.
            Although, the NN architecture is image size independent we need
            to resize images. The reason for this is, that numpy arrays have
            a fixed shape.
            :param list_IDs_temp: Indexes of the images which are part of the
                                  batch
            :return X: A 4D numpy array (batch_size, min_width, min_height, channels)
                       containing the rescaled images of the batch.
            :return y: The labels of the images as 2D matrix
                       (batch_size, number_of_classes). To create the matrix
                       the fucntion keras.utils.to_categorical is used.
        '''
        if self.use_rgb is True:
            # The function cv2.imread has an argument to read an image in RGB
            # or grayscale mode. 1 = RGB, 0 = Grayscale. Compare
            # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
            colormode = 1
            n_channels = 3
        else:
            colormode = 0
            n_channels = 1

        width_of_batch_images = []
        height_of_batch_images = []

        # We have to get the minmal image width and height of the batch.
        # Therefore, we loop over the dataframe to get width and height of
        # every image inside the batch
        for ID in list_IDs_temp:
            # Store sample
            width_of_batch_images.append(int(self.df['width'].values[ID]))
            height_of_batch_images.append(int(self.df['height'].values[ID]))

        min_width_of_batch = min(width_of_batch_images)
        min_height_of_batch = min(height_of_batch_images)

        # Tuple which will be used for cv2.rescale
        rescale_size = (min_height_of_batch, min_width_of_batch)

        # This array will be used to save the resized images. As we can see
        # the size of the array is fixed.
        X = np.empty((self.batch_size, min_width_of_batch, min_height_of_batch,
                     n_channels))
        # List to save the labels
        y = []

        for i, ID in enumerate(list_IDs_temp):
            path_to_image = self.df['path_to_image'].values[ID]
            image = cv.imread(path_to_image, colormode)
            rescaled_image = cv.resize(image, rescale_size)

            X[i, ] = rescaled_image
            y.append(self.df['race_label'].values[ID])

        return X, to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.number_IDs / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data
        :return X: A 4D numpy array (batch_size, min_width, min_height, channels)
                   containing the rescaled images of the batch.
        :return y: The labels of the images as 2D matrix
                   (batch_size, number_of_classes). To create the matrix
                   the fucntion keras.utils.to_categorical is used.
       '''

        # Generate indexes of the batch
        index_of_batch_img = self.data_index[index * self.batch_size: (index+1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(index_of_batch_img)

        return X, y
