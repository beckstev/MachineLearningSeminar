import numpy as np
from uncertainties import ufloat
import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

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


path_to_file = os.path.join(Path(os.path.abspath(__file__)).parents[0])
path_to_answers = os.path.join(path_to_file, 'answers.txt')

num_dog_races = np.genfromtxt(path_to_answers, unpack=True)

counter_num_dog_races = Counter(num_dog_races)
counter_num_dog_races_keys = list(counter_num_dog_races.keys())
counter_num_dog_races_items = list(counter_num_dog_races.values())

num_participants = len(num_dog_races)
num_total_races = 120

std_num = np.std(num_dog_races, ddof=1)
mean_num = np.mean(num_dog_races)
mean_num_dog_races = ufloat(mean_num, std_num)
mean_acc = mean_num_dog_races / num_total_races

print(f'On average a physics student at TU Dortumd university can name {mean_num_dog_races} different dog races!')
print(f'Assuming that the participants always name their known dog races correctly, the mean can be used to estimate a mean acc.')
print(f'Mean acc: {mean_acc}')

plt.figure(figsize=(5.8, 3.58))
plt.bar(counter_num_dog_races_keys, height=counter_num_dog_races_items)

x_ticks = range(1, int(max(counter_num_dog_races_keys))+2, 2)
x_ticks_strings = [str(x) for x in x_ticks]
plt.xticks(x_ticks, x_ticks_strings)

y_ticks = range(1, int(max(counter_num_dog_races_items))+2)
y_ticks_strings = [str(y) for y in y_ticks]
plt.yticks(y_ticks, y_ticks_strings)

text = f'Total number\n  of participants {num_participants}'
x_pos_text = max(counter_num_dog_races_keys) - 6
y_pos_text = max(counter_num_dog_races_items) + 0.25
plt.text(x_pos_text, y_pos_text, s=text)
plt.xlabel('Number of nameable dog races')
plt.ylabel('Number of participants')

path_to_bar_plot = os.path.join(path_to_file, 'answers_plot.pdf')
plt.savefig(path_to_bar_plot, dpi=500, pad_inches=0, bbox_inches='tight')
