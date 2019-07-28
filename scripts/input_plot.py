import pandas as pd
from pathlib import Path
import os
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
from dog_classifier.net.dataloader import DataGenerator

mpl.use('pgf')
mpl.rcParams.update(
    {'font.size': 10,
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'lualatex',
        'text.latex.unicode': True,
        'pgf.preamble': r'\DeclareMathSymbol{.}{\mathord}{letters}{"3B}',
     })


def generate_batch(batch_size):
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[1],
                                  "labels/")
    encoder_model = 'encoder_2019-06-27_19:48:46.npy'
    df_train = pd.read_csv(path_to_labels + 'train_labels.csv')
    trainDataloader = DataGenerator(df_train, encoder_model,
                                    batch_size=batch_size,
                                    use_rgb=False)
    for i in range(batch_size):
        X, y = trainDataloader.__getitem__(i)


def plot_sub(batch_size):
    path_to_file = os.path.join(Path(os.path.abspath(__file__)).parents[1],
                                "saved_models/bilder/")
    title_pad = -0.5
    fig, axes = plt.subplots(2, 3)
    fig.set_figheight(2.5)
    fig.set_figwidth(6.224)
    for i in range(batch_size):
        path_to_image = path_to_file + 'original_{}.png'.format(i)
        image = cv.imread(path_to_image, 1)
        save_img = image[..., ::-1]
        axes[0, i].set_title('Original', pad=title_pad,  fontsize=5)
        axes[0, i].imshow(save_img)
        axes[0, i].axis('off')

        path_to_image = path_to_file + 'augmented_{}.png'.format(i)
        image = cv.imread(path_to_image, 1)
        save_img = image[..., ::-1]
        axes[1, i].set_title('Augmented', pad=title_pad, fontsize=5)
        axes[1, i].imshow(save_img)
        axes[1, i].axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)
    fig.savefig(path_to_file + 'subplot.pdf', dpi=500, pad_inches=0, bbox_inches='tight')


if __name__ == '__main__':
    generate_batch(3)
    plot_sub(3)
