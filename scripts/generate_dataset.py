import argparse

from dog_classifier.io.dataset import generate_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python generate_dataset.py  <Path to label>')
    parser.add_argument('label_path', type=str, help='directory containing the labels.')
    parser.add_argument('--train', type=float, help='Training set fraction of the whole data set.')
    parser.add_argument('--test', type=float, help='Test set fraction of the whole data set.')
    parser.add_argument('--val', type=float, help='Validation set fraction of the whole data set.')
    parser.add_argument('--init', type=bool, help='Flag to add to each label file a .xml ending.')

    args = parser.parse_args()

    train = args.train if args.train else 0.6
    test = args.test if args.test else 0.2
    val = args.val if args.val else 0.2
    init = args.init if args.init else False

    generate_dataset(path_to_labels=args.label_path, train_size=train,
                     test_size=test, val_size=val, inital_run=init)
