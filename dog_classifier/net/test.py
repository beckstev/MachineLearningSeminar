import pandas as pd
from sklearn.preprocessing import LabelEncoder

from network import DogNN, SeminarNN
from dataloader import DataGenerator


if __name__ =='__main__':
    df_train = pd.read_csv('../../labels/train_labels.csv')
    encoder = LabelEncoder()
    encoder.fit(df_train['race_label'].values)
    df_train['race_label'] = encoder.transform(df_train['race_label'].values)

    trainDataloader = DataGenerator(df_train)

    model = DogNN()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

    print(type(trainDataloader))
    model.fit_generator(generator=trainDataloader, use_multiprocessing=True,
                        workers=4)
