
import argparse
from dog_classifier.evaluate import evaluate_training as eval
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python plot_history.py  <Path to saved model>')
    parser.add_argument('model_path', type=str, help='path to save/saved model')

    args = parser.parse_args()

    filepath = args.model_path + 'model_history.csv'
    df = pd.read_csv(filepath)
    eval.plot_history_special(df, args.model_path)
