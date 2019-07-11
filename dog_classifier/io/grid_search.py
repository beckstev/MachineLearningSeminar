import numpy as np
from dog_classifier.net.train import trainNN


def find_parameters(training_parameters, batch_size,  use_rgb, l2_reg):

    permutations = np.array(np.meshgrid(batch_size, use_rgb, l2_reg)).T.reshape(-1,3)

    permutations = np.array(np.meshgrid(batch_size, use_rgb, l2_reg)).T.reshape(-1, 3)
    for i in range(len(permutations)):
        try:
            use_rgb = permutations[i, 1]
            print('\n')
            print('use rgb: ', use_rgb)

            bs_size = permutations[i, 0]
            bs_size = int(bs_size)
            print('current bs: ', bs_size)

            l2_reg = permutations[i, 2]
            print('current l2_reg: ', l2_reg)
            print('\n')
            training_parameters['use_rgb'] = use_rgb
            training_parameters['batch_size'] = bs_size
            training_parameters['l2_regularisation'] = l2_reg

            trainNN(training_parameters, grid_search=True)
            
        except Exception as e:
            print(f'Error: {e} for the following perumation {permutations[i]}!')
            continue
