import numpy as np
import pandas as pd
import os
import xmltodict
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

def generate_dataset(path_to_labels,train_size, val_size, test_size, inital_run=False,):
    if inital_run ==  True:
        rename_label_files(path_to_labels)

    labels = []

    for dir in os.listdir(path_to_labels):
        path_to_dog_race_labels = os.path.join(path_to_labels, dir)

        for label_file in os.listdir(path_to_dog_race_labels):
            path_to_image_label = os.path.join(path_to_dog_race_labels,
                                               label_file)

            with open(path_to_image_label) as fd:
                doc = xmltodict.parse(fd.read())
                width = doc['annotation']['size']['width']
                height = doc['annotation']['size']['height']

                image_file = label_file.replace('.xml', '.jpg')
                path_to_image = path_to_dog_race_labels.replace('Annotation', 'Images')
                try:
                    race_label = doc['annotation']['object']['name']

                    bbox = doc['annotation']['object']['bndbox']
                    xmin, xmax = int(bbox['xmin']), int(bbox['xmax'])
                    ymin, ymax = int(bbox['ymin']), int(bbox['ymax'])

                except TypeError:
                    race_label = doc['annotation']['object'][0]['name']
                    x_cor = []
                    y_cor = []

                    for label in doc['annotation']['object']:
                        x_cor.append(int(label['bndbox']['xmin']))
                        x_cor.append(int(label['bndbox']['xmax']))
                        y_cor.append(int(label['bndbox']['ymin']))
                        y_cor.append(int(label['bndbox']['ymax']))

                    xmin, xmax = min(x_cor), max(x_cor)
                    ymin, ymax = min(y_cor), max(y_cor)
                    x = [xmin, xmin, xmax, xmax]
                    y = [ymin, ymax, ymin, ymax]

                x = [xmin, xmin, xmax, xmax]
                y = [ymin, ymax, ymin, ymax]

                corners = np.array([x, y])
                corners = sort_corners(corners)
                dog_labels = corners.flatten().tolist()

                dog_labels.extend([race_label, width, height,
                                   path_to_image, image_file])

                labels.append(dog_labels)

    columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "race_label",
               "width", "height", "path_to_label", "filename"]

    X = pd.DataFrame(data=labels, columns=columns, dtype='uint16')

    X_train_val, X_test = train_test_split(X, train_size=train_size + val_size,
                                           test_size=test_size,
                                           shuffle=True,
                                           random_state=13)

    norm = (1 - (train_size + val_size)) / 2
    train_new = train_size + norm
    val_new = val_size + norm

    X_train, X_val = train_test_split(X_train_val, train_size=train_new,
                                      test_size=val_new,
                                      shuffle=True,
                                      random_state=13)

    save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                             "labels/")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    X_train.to_csv(save_path + 'train_labels.csv', index=False)
    X_val.to_csv(save_path + 'val_labels.csv', index=False)
    X_test.to_csv(save_path + 'test_labels.csv', index=False)


def sort_corners(corners):
    # first sort upper lower
    index_upper = np.argsort(corners[1])
    corners = corners[..., index_upper]

    # second sort left right
    index_x_upper = np.argsort(corners[0][:2])
    index_x_lower = np.argsort(corners[0][2:])

    # +2 because the positions of the lower corners are
    # at index position 2 and 3
    index_x = np.append(index_x_upper, index_x_lower + 2)

    # sorted corners
    corners = corners.T[index_x]
    return corners


def rename_label_files(path_to_labels):
    ''' This function add the file ending .xml to all label files '''

    for dir in os.listdir(path_to_labels):
        path_to_dog_race_labels = os.path.join(path_to_labels, dir)

        for label_file in os.listdir(path_to_dog_race_labels):
            path_to_image_label = os.path.join(path_to_dog_race_labels,
                                               label_file)

            if '.xml' not in path_to_image_label:
                os.rename(path_to_image_label, path_to_image_label + '.xml')
            elif '.xml.xml' in path_to_image_label:
                new_path_to_image_label = path_to_image_label.replace('.xml', '')
                os.rename(path_to_image_label, new_path_to_image_label + '.xml')




if __name__ == '__main__':
    generate_dataset('../../dataset/Annotation', 0.7, 0.2, 0.1)
