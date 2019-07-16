import argparse

from dog_classifier.net.train import trainNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python main_analysis <architecure> <encoder_model>')
    parser.add_argument('architecture', type=str, help='Class name of the network')
    parser.add_argument('encoder_model', type=str, help='Name of the saved encoder model')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-l2', '--regularisation_rate', type=float)
    parser.add_argument('-p', '--early_stopping_patience', type=int)
    parser.add_argument('-d', '--early_stopping_delta', type=float)
    parser.add_argument('-ir', '--imgage_resize', type=int, nargs=2, help='Tuple (width, height) with determines the shape of the resized images')
    parser.add_argument('-n', '--n_classes', type=int, help='Number of classes to train. Default is 120')
    parser.add_argument('--use_rgb', action='store_true')

    args = parser.parse_args()

    n_epochs = int(1e2)
    if args.epochs:
        n_epochs = args.epochs

    learning_rate = 1e-3
    if args.learning_rate:
        learning_rate = args.learning_rate

    bs_size = 16
    if args.batch_size:
        bs_size = args.batch_size

    l2_reg = 0.01
    if args.regularisation_rate:
        l2_reg = args.regularisation_rate

    early_stopping_patience = 7
    if args.early_stopping_patience:
        early_stopping_patience = args.early_stopping_patience

    early_stopping_delta = 1e-4
    if args.early_stopping_delta:
        early_stopping_delta = args.early_stopping_delta

    n_classes = 120
    if args.n_classes:
        n_classes = args.n_classes

    img_resize = tuple(args.imgage_resize) if args.imgage_resize else None

    training_parameters = {'n_classes': n_classes,
                           'batch_size': bs_size,
                           'learning_rate': learning_rate,
                           'l2_regularisation': l2_reg,
                           'n_epochs': n_epochs,
                           'architecture': args.architecture,
                           'use_rgb': args.use_rgb,
                           'encoder_model': args.encoder_model,
                           'early_stopping_patience': early_stopping_patience,
                           'early_stopping_delta': early_stopping_delta,
                           'img_resize': img_resize
                           }

    trainNN(training_parameters)
