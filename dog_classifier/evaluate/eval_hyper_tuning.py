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

def exp_str(string):
    print(string)
    return np.exp(float(string))


def create_3d_subplot(df, score, figure, position, polar, azimut):
    bs = df['batch_size'].values
    l2 = df['l2_regularisation'].values
    lr = df['learning_rate'].values

    mask_max = (score == max(score))
    ax = figure.add_subplot(position, projection='3d')
    #ax = figure.add_plot(projection='3d')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Log(L2 regulation)', labelpad=5)
    ax.set_zlabel('Log(Learning rate)', labelpad=9)

    scatter_plot = ax.scatter(xs=bs[~mask_max], ys=np.log(l2[~mask_max]), zs=np.log(lr[~mask_max]),
                              c=score[~mask_max])

    ax.scatter(xs=bs[mask_max], ys=np.log(l2[mask_max]), zs=np.log(lr[mask_max]),
               c='r', marker='*', label=f'Acc: {score[mask_max][0]:.2}')

    # yticks = ax.get_yticks()
    # ylabels = [f'{exp_str(tick):.2e}' for tick in yticks]
    # ax.set_yticklabels(ylabels)

    # zticks = ax.get_zticks()
    # zlabels = [f'{exp_str(tick):.2e}' for tick in zticks]
    # ax.set_zticklabels(zlabels)

    ax.view_init(azimut, polar)
    return ax, scatter_plot


def eval_3d(df, save_path, score):
    if score is 'val_acc':
        cbar_label = 'Validation accuracy'
        sc = df[score].values

    bs = df['batch_size'].values
    l2 = df['l2_regularisation'].values
    lr = df['learning_rate'].values
    mask_max = (sc == max(sc))
    best_hyp = (int(bs[mask_max][0]), l2[mask_max][0], lr[mask_max][0])

    fig = plt.figure(figsize=(7.2, 4.45))

    ax0, scatter_plot = create_3d_subplot(df, sc, fig, 121, 64, 20)
    ax1, _ = create_3d_subplot(df, sc, fig, 122, -64, 20)
    # ax2, _ = create_3d_subplot(df, sc, fig, 133, 80, 45)

    colorbar_ax = fig.add_axes([0.22, 0.9, 0.6, 0.02])
    cbar = fig.colorbar(scatter_plot, cax=colorbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label)
    ax1.legend(loc='center left',
               bbox_to_anchor=(-0.34, 0.38, 0.2, 0.2),
               frameon=False)

    fig.suptitle(f'Optimal hyperparameter: Bs={best_hyp[0]}, L2-reg={best_hyp[1]}, Lr={best_hyp[2]}')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'hyper_raum.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, pad_inches=0.1)

if __name__ =='__main__':
    path = "../../saved_models/MiniDogNN"
    df = read_tuning_results(path)
    save_path = os.path.join(path, 'hyper_param_tuning/eval')
    eval_3d(df, save_path, 'val_acc')
