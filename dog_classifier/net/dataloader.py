import numpy as np
import cv2 as cv
from keras.utils import to_categorical, Sequence
from dog_classifier.io.transform import apply_affine_transform
from sklearn.preprocessing import LabelEncoder
from dog_classifier.io.data_augmentation import crop_range
import os
from pathlib import Path
# import matplotlib.pyplot as plt


class DataGenerator(Sequence):
    ''' Dataloader for batch wise traning of a NN with batch wise rescaling of
        images. The function inherits the properties of keras.utils.Sequence
        so that we can leverage nice functionalities such as multiprocessing.
    '''

    def __init__(self, df, encoder_model, batch_size=16, n_classes=120,
                 const_img_resize=None, use_rgb=True, shuffle=True,
                 is_test=False, is_autoencoder=False, seed=13):
        '''Initialization
        :param df: Dataframe of the dataset (training, validation or testing)
                   which contains for all images the path to the image, the
                   label and the width & height of the image
        :param batch_size:  Size of the batch
        :param n_classes: Number of classes you wanna classify
        :param const_img_resize: Tuple (width, height) with determines the
                                 shape of the resized images. If this argument
                                 is None the image will be resized batchwise
        :param use_rgb: Boolean to indicate if the function should use RGB
                        or grayscale images
        :param shuffle: To get a new image order every epoch abs
        :param is_test: Boolean to indicate if we are loading testing data. If
                         so we do not wanna use any data augmentation
        :param is_autoencoder: Boolean to indicate if the function
                               will be used to train an Autoencoder. If so
                               the y is equal to X
        :param seed: Set a seed for numpy random functions
        '''
        self.df = df
        self.batch_size = batch_size
        self.encoder_model = encoder_model
        self.number_IDs = len(df)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_rgb = use_rgb
        self.seed = seed
        self.is_test = is_test
        self.const_img_resize = const_img_resize
        self.is_autoencoder = is_autoencoder
        self.on_epoch_end()
        self.encode_labels()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.data_index = np.arange(self.number_IDs)

        if self.shuffle is True:
            np.random.seed(self.seed)
            np.random.shuffle(self.data_index)

    def encode_labels(self):
        'Encode the race into a number'
        encoder = LabelEncoder()
        path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2], "labels/")

        encoder.classes_ = np.load(path_to_labels + self.encoder_model)
        self.df['race_label'] = encoder.transform(self.df['race_label'].values)


    def batch_resize(self, list_IDs_temp):
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
        # cv2.rescale takes first the horizontal and then the vertical axis
        rescale_size = (min_width_of_batch, min_height_of_batch)
        return rescale_size


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

        if self.const_img_resize is None:
            rescale_size = self.batch_resize(list_IDs_temp)
        else:
            rescale_size = self.const_img_resize

        # This array will be used to save the resized images. As we can see
        # the size of the array is fixed.
        X = np.empty((self.batch_size, rescale_size[1], rescale_size[0],
                     n_channels))
        # List to save the labels
        y = []

        for i, ID in enumerate(list_IDs_temp):
            path_to_image = self.df['path_to_image'].values[ID]

            image = cv.imread(path_to_image, colormode) * 1/255
            # save_img = image[..., ::-1]
            # plt.imshow(save_img)
            # plt.axis('off')
            # plt.savefig('../saved_models/bilder/original_{}.png'.format(i), dpi=500, pad_inches=0, bbox_inches='tight')
            # plt.clf()

            rescaled_image = cv.resize(image, rescale_size)

            if self.is_test is False:
                # get bboxes
                bbox = np.array(self.df.loc[ID, "x1":"y4"].values, dtype='float32')
                # generate translation and zoom limits from crop_range
                trans_limits, zoom_limits = crop_range(image.shape, bbox,
                                                       rescale_size)

                # get random rotation
                random_rotation = np.random.uniform(-30, 30)
                if np.random.uniform(0, 1) < 0.5:
                    zx = zoom_limits[0]
                    zy = zoom_limits[1]
                    tx = 0
                    ty = 0
                    # text = "zoom x: {:1.2f}, zoom y: {:1.2f} \n rot: {:1.2f}".format(zx, zy, random_rotation)
                else:
                    zx = 1
                    zy = 1
                    tx = trans_limits[0]
                    ty = trans_limits[1]
                    # text = "trans x: {:1.2f}, trans y: {:1.2f} \n rot: {:1.2f}".format(tx, ty, random_rotation)
                # get random zoom fpr x and y
                # zoom_x = np.random.uniform(zoom_limits[0], 1)
                # zoom_y = np.random.uniform(zoom_limits[1], 1)

                # transform the image

                rescaled_image = apply_affine_transform(rescaled_image,
                                                        zx=zx,
                                                        zy=zy,
                                                        theta=random_rotation,
                                                        fill_mode='constant',
                                                        tx=tx,
                                                        ty=ty)
            X[i, ] = rescaled_image

            # save_img = rescaled_image[..., ::-1]
            # plt.imshow(save_img)
            # plt.axis('off')
            # plt.savefig('../saved_models/bilder/augmented_{}.png'.format(i), dpi=500, pad_inches=0, bbox_inches='tight')
            # plt.clf()

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
                   the function keras.utils.to_categorical is used.
       '''

        # Generate indexes of the batch
        index_of_batch_img = self.data_index[index * self.batch_size: (index+1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(index_of_batch_img)

        if self.is_autoencoder is False:
            return X, y
        else:
            return X, X
