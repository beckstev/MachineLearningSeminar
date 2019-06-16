import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from dog_classifier.net.dataloader import DataGenerator

model = load_model('../saved_models/our_first_nn.h5')

df_test = pd.read_csv('../labels/test_labels.csv')


# df_test = df_test.iloc[:32, :]

# Encoder for labels -> convert strings to binary classes
encoder = LabelEncoder()
encoder.fit(df_test['race_label'].values)
df_test['race_label'] = encoder.transform(df_test['race_label'].values)


testDataloader = DataGenerator(df_test, shuffle=True)

# Test predicten
Y_pred = model.predict_generator(testDataloader, verbose=1)

np.savetxt('build/prediction.txt', Y_pred)
