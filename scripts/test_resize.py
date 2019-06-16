import pandas as pd
import os
from dog_classifier.net.network import SeminarNN
from dog_classifier.net.resize_dataframe import resize_training

pfad = "build"
if not os.path.exists(pfad):
    os.makedirs(pfad)

df_train = pd.read_csv('../labels/train_labels.csv')
df_val = pd.read_csv('../labels/val_labels.csv')

train_generator, \
    val_generator, \
    STEP_SIZE_TRAIN, STEP_SIZE_VALID = resize_training(df_train, df_val, 50)

model = SeminarNN()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# model.fit_generator(generator=train_generator,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=val_generator,
#                     validation_steps=STEP_SIZE_VALID,
#                     epochs=10)

model.save('build/my_model.h5')
