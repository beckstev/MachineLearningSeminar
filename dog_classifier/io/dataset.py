import numpy as np
import pandas as pd
import os
import xmltodict
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

def generate_dataset(path_to_labels, train_size, val_size, test_size, inital_run=False,):
    ''' This function splits the stanford dataset into a training, validation
        and testing datset. The size of each dataset is specified by the params
        train_size, val_size and test_size. Furthermore, this function corrects
        a uncleanliness of the standford dataset: some of the label files do not
        have the file binding .xml. Therefore, it uses the function
        rename_label_files. This function is only required for the first initial
        call of generate_dataset - inital_run keyargument. Sometimes multiple
        dogs are in a single image, resulting in multiple boundix boxes (bbox)
        in a label file. However, the current NN-architecture is not able to
        handle multiple bboxes per image. Hence, the function creates a new bbox
        which includes all dogs together.

        :param path_to_labels: Path to label files
        :param test_size: Percentage of the dataset which will be included into
                     the test dataset
        :param val_size: Percentage of the dataset which will be included into
                    the validation dataset
        :param test_size: Percentage of the dataset which will be included into
                     the training dataset
        :param inital_run: Bollian to indicates if this is the inital run of
                           generate dataset
    '''

    if inital_run is True:
        rename_label_files(path_to_labels)

    # list which will include the label of every image
    labels = []

    # The label of each image is catorgarized into the individual dog races.
    # Hence, we have to loop over all directories inside the Annotation folder

    for dir in os.listdir(path_to_labels):

        path_to_dog_race_labels = os.path.join(path_to_labels, dir)

        # Each directories include a variety of label files (.xml) which
        # includes the dog race and bbox of every image.
        for label_file in os.listdir(path_to_dog_race_labels):

            path_to_image_label = os.path.join(path_to_dog_race_labels,
                                               label_file)

            # We open .xml file and convert the content into a dict using the
            # function xmltodict
            with open(path_to_image_label) as fd:
                doc = xmltodict.parse(fd.read())

                width = doc['annotation']['size']['width']
                height = doc['annotation']['size']['height']

                # To load the images during the traning we need the image path
                # Fortunatley, the label and image path only differ in the
                # file binding and root directory ('Annotation' <-> 'Images')
                image_file = label_file.replace('.xml', '.jpg')
                path_to_image = path_to_dog_race_labels.replace('Annotation',
                                                                'Images')
                path_to_image = path_to_image.replace('../../', '../')
                path_to_image = os.path.join(path_to_image, image_file)
                # In the following we wanna extract the dog race and bbox
                # As mentioned at the top some images include several dogs and
                # therefore several labels. This leads to an readout error
                # during the extraction process, beceause the
                # doc['annotation']['object'] is in this case a list and not a
                # dict.

                try:
                    race_label = doc['annotation']['object']['name']

                    bbox = doc['annotation']['object']['bndbox']
                    xmin, xmax = int(bbox['xmin']), int(bbox['xmax'])
                    ymin, ymax = int(bbox['ymin']), int(bbox['ymax'])

                except TypeError:
                    race_label = doc['annotation']['object'][0]['name']
                    # The xmin/xmax and ymin/xmax values of all bounding boxes
                    # will be saved to the following lists
                    x_cor = []
                    y_cor = []

                    # To get the bbox which surounds all dogs we need the max
                    # and min value of all possible boundig boxes.
                    for label in doc['annotation']['object']:
                        x_cor.append(int(label['bndbox']['xmin']))
                        x_cor.append(int(label['bndbox']['xmax']))
                        y_cor.append(int(label['bndbox']['ymin']))
                        y_cor.append(int(label['bndbox']['ymax']))

                    xmin, xmax = min(x_cor), max(x_cor)
                    ymin, ymax = min(y_cor), max(y_cor)

                # We want to transform the min and max values to
                # real image coordinates
                x = [xmin, xmin, xmax, xmax]
                y = [ymin, ymax, ymin, ymax]
                corners = np.array([x, y])

                # To make sure that we have the same coordinates order for
                # every bbox we use the function sort_corners. The function
                # sorts into the following order:
                # (x1, y1): top left corner
                # (x2, y2): top right corner
                # (x3, y3): bottom right corner
                # (x4, y4): bottom left corner
                corners = sort_corners(corners)
                # The NN only can input and output one dim objects, therefore,
                # we have to flatten our two dim array.
                dog_labels = corners.flatten().tolist()

                # We add the other lael to the bboox coordinates to
                # save all of them together into a DataFrame.
                dog_labels.extend([race_label, width, height,
                                   path_to_image, image_file])

                # We append the label of this specific image to the list with
                # all the labels
                labels.append(dog_labels)

    columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "race_label",
               "width", "height", "path_to_image", "filename"]
    # Save all the labels to a Dataframe
    df_label = pd.DataFrame(data=labels, columns=columns, dtype='uint16')

    df_train_val, df_test = train_test_split(df_label, train_size=train_size + val_size,
                                             test_size=test_size,
                                             shuffle=True,
                                             random_state=13)

    # We need to normalize the percentage amount of train and val, because
    # the parts train_size and val_size are for the whole dataset and not for
    # the dataset df_train_val (= df_label - df_test).
    norm = (1 - (train_size + val_size)) / 2
    train_new = train_size + norm
    val_new = val_size + norm

    df_train, df_val = train_test_split(df_train_val, train_size=train_new,
                                        test_size=val_new,
                                        shuffle=True,
                                        random_state=13)

    save_path = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                             "labels/")

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    df_train.to_csv(save_path + 'train_labels.csv', index=False)
    df_val.to_csv(save_path + 'val_labels.csv', index=False)
    df_test.to_csv(save_path + 'test_labels.csv', index=False)


def sort_corners(corners):
    ''' To make sure that we have the same coordinates order for
        every bbox we use the function sort_corners. The function
        sorts into the following order:

        (x1, y1): top left corner
        (x2, y2): top right corner
        (x3, y3): bottom right corner
        (x4, y4): bottom left corner

        :param corners: Numpy array with contains the bbox coordinates
        :return corners: Returns the sorted input array
    '''
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

    # Going thru all dog races directories
    for dir in os.listdir(path_to_labels):
        path_to_dog_race_labels = os.path.join(path_to_labels, dir)

        # Renameing all label file with in a dog race directories
        for label_file in os.listdir(path_to_dog_race_labels):
            path_to_image_label = os.path.join(path_to_dog_race_labels,
                                               label_file)

            # If conditon to prevent renamingfiles which already
            # including a '.xml' ending
            if '.xml' not in path_to_image_label:
                os.rename(path_to_image_label, path_to_image_label + '.xml')

            # This block just fixes double file endings and is because of the
            # first if conditon out-commented

            #elif '.xml.xml' in path_to_image_label:
            #    new_path_to_image_label = path_to_image_label.replace('.xml', '')
            #    os.rename(path_to_image_label, new_path_to_image_label + '.xml')




if __name__ == '__main__':
    generate_dataset('../../dataset/Annotation', 0.7, 0.2, 0.1)
