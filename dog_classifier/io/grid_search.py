import numpy as np
from dog_classifier.net.train import trainNN


def find_parameters(training_parameters, batch_size, learning_rate, l2_reg):

    permutations = np.array(np.meshgrid(batch_size, learning_rate, l2_reg)).T.reshape(-1,3)

    for i in range(len(permutations)):
        learning_rate = permutations[i, 1]
        print(learning_rate)

        bs_size = permutations[i, 0]
        bs_size = int(bs_size)
        print(bs_size)

        l2_reg = permutations[i, 2]
        print(l2_reg)
        training_parameters['learning_rate'] = learning_rate
        training_parameters['batch_size'] = bs_size
        training_parameters['l2_regularisation'] = l2_reg
        print('grid_search', training_parameters['batch_size'])

        trainNN(training_parameters, grid_search=True)
