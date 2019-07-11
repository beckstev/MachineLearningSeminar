import argparse

from dog_classifier.evaluate import eval_hyper_tuning as eval_ht

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python evaluate_hyper_tuning.py <model path>')
    parser.add_argument('model_path', type=str, help='Path to model')

    args = parser.parse_args()

    eval_ht.eval_ht(args.model_path)
