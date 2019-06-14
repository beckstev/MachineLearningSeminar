import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from network import DogNN, SeminarNN

df_train = pd.read_csv('../../labels/train_labels.csv')
df_val = pd.read_csv('../../labels/val_labels.csv')

# print(min(df_train['height']), min(df_train['width']))
# print(df_train['path_to_image'])
# print(df_train['filename'])

min_height_train = min(df_train['height'])
min_width_train = min(df_train['width'])

min_height_val = min(df_val['height'])
min_width_val = min(df_val['width'])

min_height = min(min_height_train, min_height_val)
min_width = min(min_width_train, min_width_val)

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_dataframe(dataframe=df_train,
                                              directory='.',
                                              x_col='path_to_image',
                                              y_col='race_label',
                                              target_size=(min_height, min_width),
                                              batch_size=50
                                              )
val_generator = datagen.flow_from_dataframe(dataframe=df_val,
                                            directory='.',
                                            x_col='path_to_image',
                                            y_col='race_label',
                                            target_size=(min_height, min_width),
                                            batch_size=50
                                            )

model = SeminarNN()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# print(model.summary())

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = val_generator.n//val_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)
