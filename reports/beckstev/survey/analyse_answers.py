import numpy as np
from uncertainties import ufloat
import os
from pathlib import Path

path_to_answers = os.path.join(Path(os.path.abspath(__file__)).parents[0],
                               'answers.txt')

num_dog_races = np.genfromtxt(path_to_answers, unpack=True)
num_participants = len(num_dog_races)
num_total_races = 120

mean_num = np.mean(num_dog_races)
std_num = np.std(num_dog_races, ddof=1)

mean_num_dog_races = ufloat(mean_num, std_num)

print(f'On average a physics student at TU Dortumd university can name {mean_num_dog_races} different dog races!')

mean_acc = mean_num_dog_races / num_total_races

print(f'Assuming that the participants always name their known dog races correctly, the mean can be used to estimate a mean acc.')
print(f'Mean acc: {mean_acc}')
