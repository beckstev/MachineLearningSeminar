import numpy as np
import pandas as pd
import os
import xmltodict
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from collections import Counter

mpl.use('pgf')
mpl.rcParams.update(
    {'font.size': 10,
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'lualatex',
        'text.latex.unicode': True,
        'pgf.preamble': r'\DeclareMathSymbol{.}{\mathord}{letters}{"3B}',
     })

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


def get_complete_df(path_to_labels):
    df_train = pd.read_csv(path_to_labels + 'train_labels.csv')
    df_val = pd.read_csv(path_to_labels + 'val_labels.csv')
    df_test = pd.read_csv(path_to_labels + 'test_labels.csv')

    df_all = pd.concat([df_train, df_val, df_test])

    return df_all


def get_height_width_dist(path_to_labels, save_path_results):
    df_all = get_complete_df(path_to_labels)

    height = df_all['height'].values
    width = df_all['width'].values

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.224, 2))

    mask_height = height < 224
    mask_width = width < 244

    print(f'Smaller in height and width: {sum((mask_height) | (mask_width))}')

    im = ax1.hist2d(height, width, bins=20, range=[[0, 600], [0, 800]])
    max_bin_content = np.amax(im[0])
    max_bin = np.where(max_bin_content == im[0])

    # max_bin includes a tuple of arrays (arrayA, arrayB)
    bin_x = max_bin[0][0]
    bin_y = max_bin[1][0]

    most_height = im[1][bin_x]
    most_width = im[2][bin_y]
    ax1.text(most_width-32, most_height+60,
            f'{int(most_width)} x {int(most_height)}', color='C8')
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Width')
    fig.colorbar(im[3], ax=ax1, label='Number of images')

    ax0.scatter(height, width, s=0.5)
    ax0.set_xlabel('Height')
    ax0.set_ylabel('Width')

    fig.suptitle(f'Number of images: {int(df_all.shape[0])}, Min Width: {min(width)}, Min height: {min(height)}')


    save_path_img = os.path.join(save_path_results, 'width_heigt_scatter_hist2d.pdf')
    plt.savefig(save_path_img,
                bbox_inches='tight', pad_inches=0)


def dog_image_cluster(path_to_labels, save_path_results, img_num_row, img_num_col):
    np.random.seed(15)
    df_all = get_complete_df(path_to_labels)

    total_number_of_img = img_num_row * img_num_col
    img_index = np.random.randint(low=0,
                                  high=df_all.shape[0],
                                  size=total_number_of_img)

    img_paths = df_all['path_to_image'].values
    img_paths = img_paths[img_index]
    fig, axes = plt.subplots(img_num_col, img_num_row, figsize=(6.224, 2))

    for ax, img_path in tqdm(zip(axes.reshape(-1), img_paths)):
        img_path = img_path.replace('..', '../..')
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    save_path_img = os.path.join(save_path_results, 'image_cluster.pdf')
    plt.savefig(save_path_img,
                bbox_inches='tight', pad_inches=0, dpi=1200)


def num_img_distribution(path_to_labels, save_path_results):
    df_all = get_complete_df(path_to_labels)

    race_labels = df_all['race_label'].values
    race_distribution = Counter(race_labels)
    num_of_img_race = np.array(list(race_distribution.values()))

    max_index = np.where(num_of_img_race == max(num_of_img_race))[0][0]
    dog_race_of_max_img = list(race_distribution.keys())[max_index]
    dog_race_of_max_img = dog_race_of_max_img.replace('_', ' ')
    num_mean = np.mean(num_of_img_race)
    print(f'Mean number of imgages: {num_mean}')
    race_placeholder = range(1, len(num_of_img_race)+1, 1)

    plt.figure(figsize=(6.224, 2))
    plt.bar(race_placeholder, num_of_img_race)
    x_y_annotate = (race_placeholder[max_index],
                    num_of_img_race[max_index])

    x_y_annotate_text = (race_placeholder[max_index] * 2/3,
                         num_of_img_race[max_index] + -20)
    plt.annotate(dog_race_of_max_img,
                 xy=x_y_annotate,
                 xytext=x_y_annotate_text,
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="angle3,angleA=0,angleB=45",
                                 color='C7'),
                c='C7')

    plt.axhline(num_mean, c='C1', alpha=0.75, linewidth=1., label='Mean')
    plt.xlabel('Race placeholder')
    plt.ylabel('Number of images')
    plt.legend(prop={'size': 6}, framealpha=0.3)
    plt.xlim(-1, 121)
    save_path_img = os.path.join(save_path_results, 'image_distribution.pdf')
    plt.savefig(save_path_img,
                bbox_inches='tight', pad_inches=0, dpi=500)




if __name__ == '__main__':
    path_to_labels = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "labels/")

    save_path_results = os.path.join(Path(os.path.abspath(__file__)).parents[2],
                                  "final_data",
                                  "general")

    if not os.path.exists(save_path_results):
        os.makedirs(save_path_results)

    #generate_dataset('../../dataset/Annotation', 0.7, 0.2, 0.1)
    get_height_width_dist(path_to_labels, save_path_results)
    dog_image_cluster(path_to_labels, save_path_results, 6, 4)
    num_img_distribution(path_to_labels, save_path_results)
