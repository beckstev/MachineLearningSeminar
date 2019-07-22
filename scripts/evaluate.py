import os
import argparse
from dog_classifier.evaluate import evaluate_training as eval
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python evaluate.py  <Path to Prediciton>, <Path to Encoder>')
    parser.add_argument('model_path', type=str, help='path to save/saved model')
    parser.add_argument('encoder_model', type=str, help='encoder model to use')
    parser.add_argument('--fname_pred', type=str, help='file name of prediction file')
    parser.add_argument('-ir', '--imgage_resize', type=int, nargs=2, help='Tuple (width, height) with determines the shape of the resized images')
    parser.add_argument('--n', type=int, help='Number of classes. Default is 120')
    parser.add_argument('--use_rgb', action='store_true')

    args = parser.parse_args()
    fname_pred = args.fname_pred if args.fname_pred else 'prediction.txt'
    n_classes = args.n if args.n else 120

    img_resize = tuple(args.imgage_resize) if args.imgage_resize else None

    if not os.path.isfile(args.model_path + '/prediction.txt'):
        eval.predict(args.model_path, args.encoder_model,
                     fname_pred, img_resize, args.use_rgb)

    Y_pred, Y_test, Y_cls, Y_true, path_to_images = eval.preprocess(args.model_path,
                                                    args.encoder_model,
                                                    fname_pred)

    # visualize_predictions
    eval.visualize_predictions(Y_pred, Y_true, path_to_images, args.encoder_model)
    # Multiclass-Analyse
    # eval.prob_multiclass(Y_pred, Y_test, Y_true, label=1, path=args.model_path)

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_cls)

    # plot the confusion matrix
    eval.plot_confusion_matrix(confusion_mtx, classes=range(n_classes),
                               path=args.model_path,
                               encoder_model=args.encoder_model)
