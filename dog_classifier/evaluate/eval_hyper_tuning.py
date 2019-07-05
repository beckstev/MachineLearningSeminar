from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os


def read_jsons(df, paths_to_json):
    params = dict()

    for path in paths_to_json:
        if os.path.isfile(path):
            with open(path, 'r') as jfile:
                param = json.load(jfile)
                # Explanation
                # https://www.geeksforgeeks.org/python-merging-two-dictionaries/
                params = {**params, **param}
    # We have to make sure that we are not adding an empty dict to params.
    # Otherwise we add NaN values to our dataframe
    if params:
        df = df.append(params, ignore_index=True)

    return df


def read_tuning_results(path_to_model):
    hyper_param_save_dict = 'hyper_param_tuning'
    path_to_model_hp = os.path.join(path_to_model, hyper_param_save_dict)

    df_param = pd.DataFrame()
    for dir in os.listdir(path_to_model_hp):

        json_hyp_param = 'loss_acc.json'
        path_hyp_param = os.path.join(path_to_model_hp, dir, json_hyp_param)

        json_traning_param = 'training_parameters.json'
        path_train_param = os.path.join(path_to_model_hp, dir, json_traning_param)

        paths_to_params = [path_train_param, path_hyp_param]
        df_param = read_jsons(df_param, paths_to_params)

    return df_param

def array_to_string_list(array):
    str_list = []
    unique_elements = np.unique(array)

    for element in unique_elements:
        str_list.append(str(element))

    return str_list


def eval_3d(df, score):

    bs = df['batch_size'].values
    l2 = df['l2_regularisation'].values
    lr = df['learning_rate'].values

    if score is 'val_acc':
        sc = df[score].values

    mask_max = sc == max(sc)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('log L2 regulation')
    ax.set_zlabel('log Learning rate')

    print(array_to_string_list(bs))
    scatter_plot = ax.scatter(bs[~mask_max], np.log(l2[~mask_max]), np.log(lr[~mask_max]), c=sc[~mask_max])
    ax.scatter(bs[mask_max], np.log(l2[mask_max]), np.log(lr[mask_max]), c='r', marker='*')
    fig.colorbar(scatter_plot)


    plt.show()

if __name__ =='__main__':
    path = "/home/beckstev/Documents/MLSeminar/MachineLearningSeminar/saved_models/MiniDogNN"
    df = read_tuning_results(path)
    eval_3d(df, 'val_acc')
