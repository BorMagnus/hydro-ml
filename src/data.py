import hashlib
import os

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class Data:
    def __init__(self, data_file, datetime_variable, data=None):
        self.data_file = data_file
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )
        self.datetime_variable = datetime_variable
        if data:
            self.data = data
        else:
            self.data = self.get_csv_data()

    def get_data(self):
        return self.data

    def get_all_variables(self):
        return list(self.data.columns)

    def get_csv_data(self):
        data_path = os.path.abspath(
            os.path.join(self.data_dir, "clean_data", self.data_file)
        )
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

    def data_transformation(
        self, data, sequence_length, target_variable, columns_to_transformation=[]
    ):
        text = (
            str(columns_to_transformation)
            if columns_to_transformation
            else "Univariate"
        )
        # Create a hash from the decomposition parameters to ensure a unique file name
        param_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        # Modify the following line to include the desired file name format
        file_name = (
            f"{os.path.splitext(self.data_file)[0]}_{sequence_length}_{param_hash}.csv"
        )
        # Modify the following line to include the 'transformation_data' directory in the file path
        decomposed_data_path = os.path.abspath(
            os.path.join(self.data_dir, "transformation_data", file_name)
        )

        if False: #os.path.isfile(decomposed_data_path): #TODO: Add way to save scalers
            lagged_df = self.load_data_from_file(decomposed_data_path)
        else:
            if columns_to_transformation:
                lagged_df = data[[target_variable] + columns_to_transformation].copy()
                # min-max normalization
                scalers = {}
                for column in lagged_df.columns:
                    scaler = MinMaxScaler()
                    lagged_df[column] = scaler.fit_transform(lagged_df[[column]])
                    scalers[column] = scaler
                # create a lagged matrix of target and variables
                new_columns = []
                for var in [target_variable] + columns_to_transformation:
                    for i in range(1, sequence_length + 1):
                        new_columns.append(lagged_df[var].shift(i).rename(f"{var}_{i}"))
                lagged_df = pd.concat([lagged_df] + new_columns, axis=1)
            else:
                lagged_df = data[[target_variable]].copy()
                # create a lagged matrix of target
                new_columns = []
                for i in range(1, sequence_length + 1):
                    new_columns.append(
                        lagged_df[target_variable]
                        .shift(i)
                        .rename(f"{target_variable}_{i}")
                    )
                lagged_df = pd.concat([lagged_df] + new_columns, axis=1)

            # remove rows with NaN values
            lagged_df.dropna(inplace=True)

            scalers = {}
            for column in lagged_df.columns:
                scaler = MinMaxScaler()
                lagged_df[column] = scaler.fit_transform(lagged_df[[column]])
                scalers[column] = scaler

            # save lagged matrix
            lagged_df.to_csv(decomposed_data_path, index=True)

        # separate the target variable from the input variables
        if columns_to_transformation:
            X = lagged_df.drop(
                columns=[target_variable] + columns_to_transformation, axis=1
            )
        else:
            X = lagged_df.drop(columns=[target_variable], axis=1)
        y = lagged_df[f"{target_variable}"]

        return X, y, scalers

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
        datetime_index = X.index
        X = torch.tensor(np.array(X)).float()
        y = torch.tensor(np.array(y)).float()

        # reshape X_train into a 3D tensor with dimensions (number of values, sequence length, number of features)
        num_values = X.shape[0]
        num_features = int(X.shape[1] / sequence_length)
        X = X.reshape(num_values, sequence_length, num_features)

        # create a PyTorch dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        # Set the datetime_index attribute
        dataloader.datetime_index = datetime_index

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
        # if round(train_size + val_size + test_size, 2) != 1.0:
        # raise ValueError(f"Train, validation, and test sizes must add up to 1.0, but is {round(train_size + val_size + test_size, 2)}") #TODO: ??? ValueError: Train, validation, and test sizes must add up to 1.0, but is 1.1 ???
        # Split the dataset into training and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        # Split the remaining data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, shuffle=False
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_datetime_values(self, indices):
        return self.data.index[indices]
    
    def inverse_transform(self, data, scalers):
        for column in data.columns:
            data[column] = scalers[column].inverse_transform(data[[column]])
        return data


    def prepare_data(
        self,
        target_variable,
        sequence_length,
        batch_size,
        variables,
        split_size,
        data=None,
    ):
        if not data:
            data = self.get_data()

        X, y, scalers = self.data_transformation(
            data=data,
            sequence_length=sequence_length,
            target_variable=target_variable,
            columns_to_transformation=variables,
        )

        train_size = split_size["train_size"]
        val_size = split_size["val_size"]
        test_size = split_size["val_size"]

        # Split the data
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(
            X, y, train_size=train_size, val_size=val_size, test_size=test_size
        )

        train_dataloader = self.create_dataloader(
            X_train, y_train, sequence_length, batch_size=batch_size, shuffle=True
        )
        val_dataloader = self.create_dataloader(
            X_val, y_val, sequence_length, batch_size=batch_size, shuffle=False
        )
        test_dataloader = self.create_dataloader(
            X_test, y_test, sequence_length, batch_size=batch_size, shuffle=False
        )

        data_loader = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }

        return data_loader, scalers
