import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def create_encoder_model():
    encode_timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    path_to_label_dic = os.path.join(Path(os.path.abspath(__file__)).parents[2], "labels/")

    df_train = pd.read_csv(path_to_label_dic + 'train_labels.csv')
    df_val = pd.read_csv(path_to_label_dic + 'val_labels.csv')
    df_test = pd.read_csv(path_to_label_dic + 'test_labels.csv')

    df_all = [df_train, df_val, df_test]
    df_all = pd.concat(df_all)

    encoder = LabelEncoder()
    encoder.fit(df_all['race_label'].values)

    np.save(path_to_label_dic + 'encoder_' + encode_timestamp + '.npy', encoder.classes_)
