import pandas as pd
from pathlib import Path
import os
import cv2 as cv
import matplotlib.pyplot as plt
from dog_classifier.net.dataloader import DataGenerator


def generate_batch(batch_size):
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[1],
                                  "labels/")
    encoder_model = 'encoder_2019-06-20_14:43:56.npy'
    df_train = pd.read_csv(path_to_labels + 'train_labels.csv')
    trainDataloader = DataGenerator(df_train, encoder_model,
                                    batch_size=batch_size)
    for i in range(batch_size):
        X, y = trainDataloader.__getitem__(i)


def plot_sub(batch_size):
    path_to_file = os.path.join(Path(os.path.abspath(__file__)).parents[1],
                                "saved_models/bilder/")

    fig, axes = plt.subplots(2, 3)
    fig.set_figheight(3.58)
    fig.set_figwidth(5.8)
    for i in range(batch_size):
        path_to_image = path_to_file + 'original_{}.png'.format(i)
        image = cv.imread(path_to_image, 1)
        save_img = image[..., ::-1]
        axes[0, i].set_title('Original')
        axes[0, i].imshow(save_img)
        axes[0, i].axis('off')

        path_to_image = path_to_file + 'augmented_{}.png'.format(i)
        image = cv.imread(path_to_image, 1)
        save_img = image[..., ::-1]
        axes[1, i].set_title('Augmented')
        axes[1, i].imshow(save_img)
        axes[1, i].axis('off')

    fig.savefig(path_to_file + 'subplot.pdf', dpi=500, pad_inches=0, bbox_inches='tight')


if __name__ == '__main__':
    generate_batch(3)
    plot_sub(3)
