import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from network import DogNN

df_train = pd.read_csv('../../labels/train_labels.csv')

model = DogNN()
