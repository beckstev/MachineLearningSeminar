import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score

from dog_classifier.net.dataloader import DataGenerator
import dog_classifier.autoencoder.randomforest as dog_rf


def eval_rf_training(rf, X, y):
    num_cv = 3
    score_cv_acc = cross_validate(rf, X, y,
                                  scoring=make_scorer(accuracy_score),
                                  cv=num_cv)
    score_acc = dict()
    score_acc['mean'] = score_cv_acc['test_score'].mean()
    score_acc['std'] = score_cv_acc['test_score'].std(ddof=1)
    print(f'Accuracy {score_acc["mean"]:0.3f} +/- {score_acc["std"]:0.3f}')

    return score_acc


def get_rf_prediction(Dataloader):
    model_save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                   "saved_models",
                                   "autoencoder")
    encoder = dog_rf.get_encoder(model_save_path)

    rf_name = 'randomforest.sav'

    path_to_rf = os.path.join(model_save_path, rf_name)
    with open(path_to_rf, 'rb') as f:
        rf = pickle.load(f)

    X_rf = encoder.predict_generator(Dataloader, verbose=1)
    y_pred = rf.predict(X_rf)

    return y_pred


def get_test_datagenerator(encoder_model, img_resize):

    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    df_test = pd.read_csv(path_to_labels + 'test_labels.csv')

    testDataloader = DataGenerator(df_test,
                                   encoder_model=encoder_model,
                                   shuffle=True,
                                   is_test=True,
                                   use_rgb=True,
                                   const_img_resize=img_resize)

    return testDataloader


def get_true_labels_and_img_paths(dataloader, y_pred):
    y = dataloader.df['race_label'].values
    y_true = y[dataloader.data_index]
    path_to_images = dataloader.df['path_to_image'].values
    path_to_images = path_to_images[dataloader.data_index]

    diff = (y_true.shape[0] - y_pred.shape[0])
    if diff is not 0:
        y_true = y_true[:-diff]
        path_to_images = path_to_images[:-diff]
    return y_true, path_to_images


def get_label_encoder(encoder_model):
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    encoder_path = os.path.join(path_to_labels, encoder_model)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_path)
    return encoder

def visualize_rf_preduction(encoder, img_resize):
    test_dataloader = get_test_datagenerator(encoder, img_resize)
    y_pred = get_rf_prediction(test_dataloader)
    y_true, img_paths = get_true_labels_and_img_paths(test_dataloader, y_pred)
    label_encoder = get_label_encoder(encoder)
    for index in range(len(y_pred)):
        race_true = label_encoder.inverse_transform((y_true[index],))[0]
        race_pred = label_encoder.inverse_transform((y_pred[index],))[0]
        img_path = img_paths[index]

        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f'True: {race_true}   -   Prediction: {race_pred}')
        plt.axis('off')
        plt.show()




if __name__ == '__main__':
    encoder = 'encoder_2019-06-27_19:48:46.npy'
    visualize_rf_preduction(encoder)
