import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch

class load_data():
    def __init__(self):
        self.sc = StandardScaler()

    def get_raw_data(self):
        path = './data/raw_data/cascaded_use_case_data.xlsx'
        if os.path.isfile(path):
            return pd.read_excel(path)
        else:
            return AssertionError("Raw data file does not exist!")
        
    def get_univariate_data(self, bottling=False):
        if bottling:
            path = './data/clean_data/univariate/Q_Kalltveit_uten_tapping/Q_Kalltveit_uten_tapping.csv'
            if os.path.isfile(path):
                return pd.read_csv(path, index_col="Datetime")
            else:
                raw_data = self.get_raw_data()
                target = raw_data[['Q_Kalltveit_uten_tapping']]
                target.index = raw_data['Datetime']
                target.to_csv(path)
                return target
        else:
            path = './data/clean_data/univariate/Q_Kalltveit/Q_Kalltveit.csv'
            if os.path.isfile(path):
                return pd.read_csv(path, index_col="Datetime")
            else:
                raw_data = self.get_raw_data()
                target = raw_data[['Q_Kalltveit']]
                target.index = raw_data['Datetime']
                target.to_csv(path)
                return target

    def get_multivariate_data(self):
        pass

    def get_spatio_temporal_data(self):
        pass

    def get_scaler(self):
        return self.sc
    
    def split_data(self, data, shuffle_train=False, train_size=0.7, test_size=0.3):
        train, temp = train_test_split(data, train_size=train_size, shuffle=shuffle_train)
        val, test = train_test_split(temp, test_size=test_size, shuffle=False)
        return train, val, test

    def sliding_windows(self, data, seq_length):
        x = []
        y = []
        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)
        return torch.from_numpy(np.array(x).reshape(len(x), -1)).float(), torch.from_numpy(np.array(y)).float()

    def get_lagged_data(self, sequence_length, pca=False):
        
        data = self.get_univariate_data()
        train, val, test = self.train_test_split(data)

        X_train, y_train = self.sliding_windows(train.values, sequence_length)
        X_val, y_val = self.sliding_windows(val.values, sequence_length)
        X_test, y_test = self.sliding_windows(test.values, sequence_length)

        if pca:
            pca = PCA(n_components = 0.95)
            pca.fit_transform(X_train)
            k = len(np.cumsum(pca.explained_variance_ratio_*100))

            X_train = X_train[:, :k]
            X_val = X_val[:, :k]
            X_test = X_test[:, :k]

        return torch.utils.data.TensorDataset(X_train, y_train), torch.utils.data.TensorDataset(X_val, y_val), torch.utils.data.TensorDataset(X_test, y_test)
