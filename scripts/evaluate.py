import argparse
import matplotlib.pyplot as plt
from dog_classifier.evaluate import evaluate_training as eval
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python evaluate.py  <Path to Prediciton>, <Path to Encoder>, <File Name of Prediction>')
    parser.add_argument('prediction_path', type=str, help='path to save/saved prediction')
    parser.add_argument('encoder_path', type=str, help='path to encoder')
    parser.add_argument('fname_pred', type=str, help='file name of prediction file')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--init', type=bool, help='Flag to predict one time.')

    args = parser.parse_args()
    init = args.init if args.init else False

    if init:
        eval.predict(args.model_path, args.prediction_path, args.encoder_path,
                     args.fname_pred)

    Y_pred, Y_test, Y_cls, Y_true = eval.preprocess(args.prediction_path +
                                                    args.fname_pred,
                                                    args.encoder_path)

    # Multiclass-Analyse
    eval.prob_multiclass(Y_pred, Y_test, label=10, path='.')

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_cls)

    # plot the confusion matrix
    # Prblem with figure size, still need fixing, some axis is cut off
    # plt.figure(figsize=(8, 8))
    # eval.plot_confusion_matrix(confusion_mtx, classes=range(120), path=".")
