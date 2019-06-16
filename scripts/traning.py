import argparse

# BillDiction Library
from dog_classifier.net.train import trainNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python main_analysis <architecure>')
    parser.add_argument('architecture', type=str, help='Class name of the network')
    parser.add_argument('encoder_model', type=str, help='Name of the saved encoder model')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('--use_rgb', action='store_true')

    args = parser.parse_args()

    n_epochs = int(1e5)
    if args.epochs:
        n_epochs = args.epochs

    learning_rate = 1e-4
    if args.learning_rate:
        learning_rate = args.learning_rate

    bs_size = 8
    if args.batch_size:
        bs_size = args.batch_size

    if args.use_rgb:
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        norm_mean, norm_std = [0.5], [0.2]

    training_parameters = {'batch_size': bs_size,
                           'learning_rate': learning_rate,
                           'n_epochs': n_epochs,
                           'architecture': args.architecture,
                           'use_rgb': args.use_rgb,
                           'encoder_model': args.encoder_model,
                           'normalization': {
                                            'mean': norm_mean,
                                            'std': norm_std}

                           }

    trainNN(training_parameters)
