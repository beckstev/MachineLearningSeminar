import argparse

from dog_classifier.autoencoder.autoencoder import train_autoencoder
from dog_classifier.autoencoder.randomforest import train_random_forest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python main_analysis <architecure> <encoder_model>')
    parser.add_argument('encoder_model', type=str, help='Name of the saved encoder model')
    parser.add_argument('image_resize', type=int, nargs=2, help=' Width & Height wich determines the shape of the resized images')
    parser.add_argument('n_classes', type=int, help='Number of classes to train.')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-p', '--early_stopping_patience', type=int)
    parser.add_argument('-d', '--early_stopping_delta', type=float)
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')

    args = parser.parse_args()

    n_epochs = int(5e1)
    if args.epochs:
        n_epochs = args.epochs

    learning_rate = 1e-3
    if args.learning_rate:
        learning_rate = args.learning_rate

    bs_size = 16
    if args.batch_size:
        bs_size = args.batch_size

    early_stopping_patience = 10
    if args.early_stopping_patience:
        early_stopping_patience = args.early_stopping_patience

    early_stopping_delta = 1e-5
    if args.early_stopping_delta:
        early_stopping_delta = args.early_stopping_delta

    even_divider = sum(args.image_resize) % 2**3

    # The image size has to be an even divider of 8. The reason for it is
    # that we are using three downsampling layers for the autoencoder

    assert even_divider is 0, 'The image size has to be an even divider of 8!'

    img_resize = tuple(args.image_resize) if args.image_resize else None

    training_parameters = {'n_classes': args.n_classes,
                           'batch_size': bs_size,
                           'learning_rate': learning_rate,
                           'n_epochs': n_epochs,
                           'use_rgb': args.use_rgb,
                           'encoder_model': args.encoder_model,
                           'early_stopping_patience': early_stopping_patience,
                           'early_stopping_delta': early_stopping_delta,
                           'img_resize': img_resize
                           }

    train_autoencoder(training_parameters)
    train_random_forest(training_parameters, grid_search=args.grid_search)
