from keras.datasets import mnist
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import importlib.util
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense

spec = importlib.util.spec_from_file_location("evaluate", "../dog_classifier/evaluate/evaluate_training.py")
eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def data_preprocessing(X_train, X_test, Y_train, Y_test):
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


# plt.figure(figsize=(12, 10))
# x, y = 10, 4
# for i in range(40):
#     plt.subplot(y, x, i+1)
#     plt.imshow(X_train[i], interpolation='nearest', cmap=cm.Greys)
# plt.show()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# print(X_train.shape)
X_train, X_test, Y_train, Y_test = data_preprocessing(X_train,
                                                      X_test, Y_train, Y_test)
print(X_train.shape)
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

history = model.fit(X_train, Y_train, batch_size=512, epochs=15,
                    verbose=1, validation_data=(X_val, Y_val))

eval.plot_history(history)
