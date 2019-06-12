import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from network import DogNN, SeminarNN

df_train = pd.read_csv('../../labels/train_labels.csv')

model = SeminarNN()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
