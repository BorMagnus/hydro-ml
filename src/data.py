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
import hashlib

# TODO: Save data_loaders?
class Data:
    def __init__(self, data_file, datetime_variable):
        self.data_file = data_file
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        self.datetime_variable = datetime_variable
        self.data = self.get_csv_data()


    def get_data(self):
        return self.data


    def get_csv_data(self):
        data_path = os.path.abspath(os.path.join(self.data_dir, "clean_data", self.data_file))
        if os.path.isfile(data_path):
            data = self.load_data_from_file(data_path)
            return data
        else:
            raise FileNotFoundError("File does not exist at path: {}".format(data_path))
        
    
    def load_data_from_file(self, data_path):
        """Loads a pandas DataFrame from a CSV file."""
        with FileLock(os.path.expanduser("~/.data.lock")):
            # get file extension
            file_ext = os.path.splitext(data_path)[1]
            # check if file is .csv
            if file_ext == ".csv":
                data = pd.read_csv(data_path, index_col=self.datetime_variable)
            else:
                raise ValueError("Invalid file format. Only CSV files are supported.")
        return data
    
    # TODO: Add check to see if columns_to_transformation is in data.
    def data_transformation(self, sequence_length, target_variable, columns_to_transformation=[]):

        text = str(columns_to_transformation) if columns_to_transformation else "Univariate"
        # Create a hash from the decomposition parameters to ensure a unique file name
        param_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        # Modify the following line to include the desired file name format
        file_name = f"{os.path.splitext(self.data_file)[0]}_{sequence_length}_{param_hash}.csv"
        # Modify the following line to include the 'transformation_data' directory in the file path
        decomposed_data_path = os.path.abspath(os.path.join(self.data_dir, "transformation_data", file_name))

        if os.path.isfile(decomposed_data_path):
            lagged_df = self.load_data_from_file(decomposed_data_path)
        else:
            data = self.get_data()
            if columns_to_transformation:
                lagged_df = data[[target_variable] + columns_to_transformation].copy()
                # create a lagged matrix of target and variables
                for var in [target_variable] + columns_to_transformation:
                    for i in range(1, sequence_length+1):
                        lagged_df.loc[:, f'{var}_{i}'] = lagged_df[var].shift(i)
            else:
                lagged_df = data[[target_variable]].copy()
                # create a lagged matrix of target
                for i in range(1, sequence_length+1):
                    lagged_df.loc[:, f'{target_variable}_{i}'] = lagged_df[target_variable].shift(i)

        # remove rows with NaN values
        lagged_df.dropna(inplace=True)

        # save lagged matrix
        lagged_df.to_csv(decomposed_data_path, index=True)

        # separate the target variable from the input variables
        if columns_to_transformation:
            X = lagged_df.drop(columns=[target_variable] + columns_to_transformation, axis=1)
        else:
            X = lagged_df.drop(columns=[target_variable], axis=1)
        y = lagged_df[f'{target_variable}']

        X = torch.tensor(np.array(X)).float()
        y = torch.tensor(np.array(y)).float()

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
        
        # reshape X_train into a 3D tensor with dimensions (number of values, sequence length, number of features)
        num_values = X.shape[0]
        num_features = int(X.shape[1]/sequence_length)
        X = X.reshape(num_values, sequence_length, num_features)

        # create a PyTorch dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


    def split_data(self, X, y, train_size=0.7, val_size=0.2, test_size=0.1):
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
        if round(train_size + val_size + test_size, 2) != 1.0:
            raise ValueError("Train, validation, and test sizes must add up to 1.0")
        # Split the dataset into training and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                                    test_size=test_size, 
                                                                    shuffle=False)
        # Split the remaining data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                        test_size=val_size,
                                                        shuffle=False)
        return X_train, y_train, X_val, y_val, X_test, y_test
    

    def prepare_data(self, target_variable, sequence_length, batch_size, variables):
        X, y = self.data_transformation(
            sequence_length=sequence_length, 
            target_variable=target_variable,
            columns_to_transformation=variables
        )

        #TODO: Posiblility to set train, val, test size
        train_size = 0.7
        val_size = 0.2
        test_size = 0.1

        # Split the data
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y, train_size=train_size, val_size=val_size, test_size=test_size)
        train_dataloader = self.create_dataloader(X_train, y_train, sequence_length, batch_size=batch_size, shuffle=True)
        val_dataloader = self.create_dataloader(X_val, y_val, sequence_length, batch_size=batch_size, shuffle=False)
        test_dataloader = self.create_dataloader(X_test, y_test, sequence_length, batch_size=batch_size, shuffle=False)
        
        data_loader = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }
        
        return data_loader