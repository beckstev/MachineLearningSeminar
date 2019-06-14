import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.utils import to_categorical
from network import DogNN, SeminarNN


class DataGenerator(Sequential):

    def __init__(self, df, batch_size=16,
                 n_channels=3, n_classes=120, use_rgb=True, shuffle=True):
                 'Initialization'
                 self.df = df
                 self.batch_size = batch_size
                 self.number_IDs = len(df)
                 self.n_classes = n_classes
                 self.shuffle = shuffle
                 self.use_rgb = use_rgb
                 self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.number_IDs)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization

        if self.use_rgb is True:
            colormode = 1
            n_channels = 3
        else:
            colormode = 0
            n_channels = 1

        width_of_batch_images = []
        height_of_batch_images = []

        for ID in list_IDs_temp:
            # Store sample
            print(ID)
            width_of_batch_images.append(int(self.df['width'].values[ID]))
            height_of_batch_images.append(int(self.df['height'].values[ID]))

        min_width_of_batch = min(width_of_batch_images)
        min_height_of_batch = min(height_of_batch_images)

        rescale_size = (min_height_of_batch, min_width_of_batch)
        X = np.empty((self.batch_size, min_width_of_batch, min_height_of_batch,
                     n_channels))
        y = np.empty(self.batch_size, dtype=int)

        for ID in list_IDs_temp:
            path_to_image = self.df['path_to_image'].values[ID]
            print(path_to_image)
            image = cv.imread(path_to_image, colormode)
            plt.imshow(image)
            plt.show()
            rescaled_image = cv.resize(image, rescale_size)
            X[i, ] = rescaled_image
            y[i] = str(self.df['race_label'].values[ID])

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.number_IDs / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1)*self.batch_size]

        list_of_indexes = np.arange(self.number_IDs)
        # Find list of IDs
        list_IDs_temp = [list_of_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

if __name__ =='__main__':
    df_train = pd.read_csv('../../labels/train_labels.csv')

    print(df_train['width'].values[0])
    trainDataloader = DataGenerator(df_train)

    X, y = trainDataloader.__getitem__(0)

    #model = DogNN()

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
