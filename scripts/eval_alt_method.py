import argparse

from dog_classifier.evaluate import evaluate_randomforest as eval_rf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python eval_alt_method.py <Encoder>')
    parser.add_argument('encoder_model', type=str, help='encoder model to use')
    parser.add_argument('image_resize', type=int, nargs=2, help=' Width & Height wich determines the shape of the resized images')
    parser.add_argument('n_classes', type=int, help='Number of classes to train.')

    args = parser.parse_args()
    img_resize = tuple(args.image_resize) if args.image_resize else None

    eval_rf.visualize_rf_preduction(args.encoder_model, img_resize, args.n_classes)
