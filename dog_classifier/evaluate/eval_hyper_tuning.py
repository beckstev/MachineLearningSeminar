import numpy as np
import matplotlib.pyplot as plt
import json
import os

def read_direct(path_to_save):
    for dir in os.listdir(path_to_save):
        json_save_file = 'loss_acc.json'
        path_to_hyp_jyson = os.path.join(path_to_save, dir, json_save_file)
        is os.isdir(path_to_hyp_jyson):
            with open(path_to_hyp_jyson, 'r') as jfile:
                params = json.load(jfile)

        print(params)




if __name__ =='__main__':
    path = "/home/beckstev/Documents/MLSeminar/MachineLearningSeminar/saved_models/MiniDogNN/hyper_param_tuning"
    read_direct(path)
