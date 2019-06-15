from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.cm as cm
# import importlib.util
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.metrics import confusion_matrix
from dog_classifier.evaluate import evaluate_training as eval
# for dropout
# from keras.layers.core import Dropout
# from keras.regularizers import l2

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def data_preprocessing(X_train, X_test, Y_train, Y_test):
    """function for data preprocessing of the mnist data set"""
    # reshape and change dtype
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # scale input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)

    return X_train, X_test, Y_train, Y_test


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test, Y_train, Y_test = data_preprocessing(X_train,
                                                      X_test, Y_train, Y_test)
# split in training and Validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=0.3,
                                                  random_state=42,
                                                  stratify=Y_train)

# Neurales Netz bauen
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Neurales Netz trainieren
history = model.fit(X_train, Y_train, batch_size=512, epochs=15,
                    verbose=1, validation_data=(X_val, Y_val))

# History plotten
eval.plot_history(history, "history")

# Test predicten
Y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors
Y_cls = np.argmax(Y_pred, axis=1)

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test, axis=1)

# Multiclass-Analyse
eval.prob_multiclass(Y_pred, Y_test, label=0, fname="multiclass")

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_cls)

# plot the confusion matrix
# Prblem with figure size, still need fixing, some axis is cut off all the time
plt.figure(figsize=(8, 8))
eval.plot_confusion_matrix(confusion_mtx, classes=range(10), fname='cm_norm')
plt.figure(figsize=(8, 8))
eval.plot_confusion_matrix(confusion_mtx, classes=range(10),
                           normalize=False, fname='cm')

# Show the top 6 errors
eval.display_errors(6, Y_cls, Y_true, Y_pred, X_test,
                    height=28, width=28, nrows=2, ncols=3, fname='error_plot')

eval.evaluate_test_params(model, X_test, Y_test)
eval.plot_predictions(12, model, X_test, 28, 28, "predictions")

# model using Dropout
# dropout = 0.5
# l2_lambda = 0.0001
#
# model_dropout = Sequential()
# model_dropout.add(Dense(512, activation='relu',
#                   kernel_regularizer=l2(l2_lambda), input_dim=784))
# model_dropout.add(Dropout(dropout))
# model_dropout.add(Dense(256, activation='relu',
#                   kernel_regularizer=l2(l2_lambda)))
# model_dropout.add(Dropout(dropout))
# model_dropout.add(Dense(10, activation='softmax'))
#
# train_hist = eval.HistoryEpoch((X_train, Y_train))
# val_hist = eval.HistoryEpoch((X_val, Y_val))
# test_hist = eval.HistoryEpoch((X_test, Y_test))
#
#
# model_dropout.compile(loss='categorical_crossentropy',
#                       optimizer='adam', metrics=['accuracy'])
#
# history_dropout = model_dropout.fit(X_train, Y_train,
#                                     batch_size=512, epochs=15, verbose=1,
#                                     validation_data=(X_val, Y_val),
#                                     callbacks=[val_hist,
#                                                train_hist, test_hist])
# eval.plot_history_epoch(train_hist, val_hist, test_hist, "history_epoch")
