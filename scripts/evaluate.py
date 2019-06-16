import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# from keras.models import load_model
from dog_classifier.evaluate import evaluate_training as eval
from dog_classifier.net.dataloader import DataGenerator
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical


# model = load_model('../saved_models/our_first_nn.h5')

df_test = pd.read_csv('../labels/test_labels.csv')
# df_test = df_test.iloc[:32, :]
testDataloader = DataGenerator(df_test, shuffle=True)
# Encoder for labels -> convert strings to binary classes
encoder = LabelEncoder()
encoder.fit(df_test['race_label'].values)
df_test['race_label'] = encoder.transform(df_test['race_label'].values)
Y_pred = np.genfromtxt('build/prediction.txt')

# Y_test erstellen, indem race_labels konvertiert werden
values = df_test['race_label'].values
test = values[testDataloader.data_index]
Y_test = to_categorical(test[:-4], num_classes=None)

# print(Y_pred.shape)
# Convert predictions classes to one hot vectors
Y_cls = np.argmax(np.array(Y_pred), axis=1)

# Convert validation observations to one hot vectors
Y_true = np.argmax(np.array(Y_test), axis=1)

# Multiclass-Analyse
eval.prob_multiclass(Y_pred, Y_test, label=10, fname="multiclass")

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_cls)

# plot the confusion matrix
# Prblem with figure size, still need fixing, some axis is cut off all the time
plt.figure(figsize=(8, 8))
eval.plot_confusion_matrix(confusion_mtx, classes=range(120), fname='cm_norm')
# plt.figure(figsize=(8, 8))
# eval.plot_confusion_matrix(confusion_mtx, classes=range(120),
                           # normalize=False, fname='cm')
