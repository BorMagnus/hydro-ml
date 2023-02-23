import os
from filelock import FileLock
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

class load_data:
    def __init__(self, data_dir, target_variable):
        # Getting the directory of the config file and giving the data path
        self.data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + data_dir # TODO: Use cwd instead (cleaner)
        self.target_variable = target_variable


    def get_raw_data(self):
        # Construct file path using os.path.join

        raw_data_path = os.path.join(self.data_dir, "raw_data", "cascaded_use_case_data.xlsx")
        
        if os.path.isfile(raw_data_path):
            return self.load_data_from_file(raw_data_path)
        else:
            raise FileNotFoundError("Raw data file does not exist at path: {}".format(raw_data_path))

        
    def get_target_data(self, target_variable, path):
        """

        """
        self.target_variable = target_variable
        if os.path.isfile(path):
            return pd.read_csv(path, index_col='Datetime')
        else:
            data = self.get_raw_data()
            target_data = data[target_variable]
            target_data.index = data['Datetime']
            target_data.to_csv(path)
            return pd.read_csv(path, index_col='Datetime')
                    

    def create_lagged_matrix(self, window_size, vars_to_lag=None, pca=False, mi=False): #TODO: Fix decimal (five values in dataframe four in X, y)
        """
        Create a lagged matrix from time series data.

        Args:
        - window_size: number of lags to include.
        - vars_to_lag: list of variable names to include in the lagged matrix.
        If None, all variables except the target variable are included.

        Returns:
        - X: tensor array of shape (n_samples, window_size, ).
        - y: tensor array of shape (n_samples,).
        """

        # Construct file path using os.path.join
        path = os.path.join(
            self.data_dir, 
            "clean_data", 
            "multivariate", 
            self.target_variable, 
            f"{window_size}_lag_" + ("_".join(vars_to_lag) if vars_to_lag else "") + ".csv")
            
        if os.path.isfile(path): 
            lagged_df = self.load_data_from_file(path)
        else:
            data = self.get_raw_data()

            if vars_to_lag:
                lagged_df = data[[self.target_variable] + vars_to_lag].copy()
                # create a lagged matrix of target and variables
                for i in range(1, window_size+1):
                    for var in vars_to_lag:
                        lagged_df.loc[:, f'{var}_{i}'] = lagged_df[var].shift(i)
            else:
                lagged_df = data[[self.target_variable]].copy()
                # create a lagged matrix of target
                for i in range(1, window_size+1):
                    lagged_df.loc[:, f'{self.target_variable}_{i}'] = lagged_df[self.target_variable].shift(i)

            # set datetime to index
            lagged_df.index = data['Datetime']

            # remove rows with NaN values
            lagged_df.dropna(inplace=True)

            # save lagged matrix
            lagged_df.to_csv(path, index=True)
        
        # separate the target variable from the input variables
        X = lagged_df.drop(columns=[f'{self.target_variable}'], axis=1)
        y = lagged_df[f'{self.target_variable}'] # TODO: methods such as Granger causality or structural equation modeling to determine whether the lagged values of "Nedbør Nilsebu" and "Q_Lyngsaana" are predictive of future values of "Q_Kalltveit".

        if pca:
            # define standard scaler
            scaler = StandardScaler()
            # Standardize the data
            X_scaled = scaler.fit_transform(X)

            # Apply PCA that will choose the minimum number of components necessary to explain 90% of the variance in the data
            pca = PCA(n_components=0.9)
            pca_features = pca.fit_transform(X_scaled)
            X = pca_features
        
        if mi: #TODO: Takes no account to structure of the features #TODO: If the relationship between "Nedbør Nilsebu", "Q_Lyngsaana", and "Q_Kalltveit" is non-linear, then linear models such as PCA and mutual information may not be sufficient to capture this relationship.
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            discrete_features = X.dtypes == int
            y = y.values.ravel() # convert y to 1-dimensional array
            mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
            mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
            mi_scores = mi_scores.sort_values(ascending=False)
            selected_dimension = mi_scores[mi_scores.values >= 1]
            X = X[selected_dimension.index].to_numpy()

        X = torch.tensor(np.array(X)).float()
        y = torch.tensor(np.array(y)).float()
        if not vars_to_lag:
            # reshape X into a 3D tensor with dimensions 
            # (number of sequences, sequence length, 1) if univariate
            X = X.unsqueeze(-1)

        return X, y

    
    def create_dataloader(self, X, y, sequence_length, batch_size, shuffle):
        """
        Creates a PyTorch DataLoader from input data X and target data y.
        
        Parameters:
            X (ndarray): The input data.
            y (ndarray): The target data.
            sequence_length (int): The length of each sequence in the input data.
            batch_size (int): The batch size to use for the DataLoader.
            shuffle (bool): Whether to shuffle the data before creating the DataLoader.
        
        Returns:
            A PyTorch DataLoader object.
        """
        if X.shape[-1] != 1:
            # reshape X_train into a 3D tensor with dimensions (number of sequences, sequence length, number of features)
            num_sequences = X.shape[0]
            num_features = X.shape[1]
            X_3d = np.zeros((num_sequences, sequence_length, num_features))
            for i in range(sequence_length, num_sequences):
                X_3d[i] = X[i-sequence_length:i, :]
            X_3d = X_3d.astype(np.float32)
            X = X_3d.copy()
            X = torch.tensor(X)

        # create a PyTorch dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    
    def split_data(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Splits the dataset into training, validation, and test sets.
        
        Parameters:
            X (array-like): The input data.
            y (array-like): The target data.
            train_size (float): The proportion of the dataset to use for training.
            val_size (float): The proportion of the dataset to use for validation.
            test_size (float): The proportion of the dataset to use for testing.
        
        Returns:
            A tuple (X_train, y_train, X_val, y_val, X_test, y_test) containing the
            training, validation, and test sets.
        """
        # Check that the sizes add up to 1.0
        if train_size + val_size + test_size != 1.0:
            raise ValueError("Train, validation, and test sizes must add up to 1.0")

        # Split the dataset into training and test sets
        X_train_test, X_test, y_train_test, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

        # Compute the validation size relative to the remaining data after the test split
        val_size_ratio = val_size / (train_size + val_size)
        
        # Split the remaining data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_test, y_train_test,
                                                        test_size=val_size_ratio,
                                                        shuffle=False)

        return X_train, y_train, X_val, y_val, X_test, y_test


    def load_data_from_file(self, file_path):
        """Loads a pandas DataFrame from a CSV or XLSX file."""
        with FileLock(os.path.expanduser("~/.data.lock")):
            # get file extension
            file_ext = os.path.splitext(file_path)[1]

            # check if file is .csv or .xlsx
            if file_ext == ".csv":
                data = pd.read_csv(file_path, index_col='Datetime')
            elif file_ext == ".xlsx":
                data = pd.read_excel(file_path)

        return data 