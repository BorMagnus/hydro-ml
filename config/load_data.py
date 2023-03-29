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
        self.data_dir = data_dir + '/data'
        self.target_variable = target_variable


    def get_raw_data(self): # TODO: Needs to change the variable names
        # Construct file path using os.path.join
        raw_data_path = os.path.join(self.data_dir, "raw_data", "cascaded_use_case_data.xlsx")
        
        if os.path.isfile(raw_data_path):
            data = self.load_data_from_file(raw_data_path)
            # Rename columns to be more user-friendly
            data = data.rename(columns={
                #TODO: Remove and rename before
                'Vindhastighet Nilsebu': 'Wind_Speed_Nilsebu',
                'Lufttemp. Nilsebu': 'Air_Temperature_Nilsebu',
                'Vindretning Nilsebu': 'Wind_Direction_Nilsebu',
                'RelHum Nilsebu': 'Relative_Humidity_Nilsebu',
                'Vannstand Lyngsana': 'Water_Level_Lyngsaana',
                'Vanntemp. Hiafossen': 'Water_Temperature_Hiafossen',
                'Vannstand Hiafossen': 'Water_Level_Hiafossen',
                'Lufttemp Fister': 'Air_Temperature_Fister',
                'Nedbør Fister': 'Precipitation_Fister',
                'Q_Lyngsvatn_overlop': 'Flow_Lyngsvatn_Overflow',
                'Q_tapping': 'Flow_Tapping',
                'Vannstand Kalltveit': 'Water_Level_Kalltveit',
                'Q_Kalltveit': 'Flow_Kalltveit',
                'Vanntemp. Kalltveit kum': 'Water_Temperature_Kalltveit_Kum',
                'Nedbør Nilsebu': 'Precipitation_Nilsebu',
                'Vanntemp. Hiavatn': 'Water_Temperature_Hiavatn',
                'Vannstand Hiavatn': 'Water_Level_Hiavatn',
                'Vanntemp. Musdalsvatn': 'Water_Temperature_Musdalsvatn',
                'Vannstand Musdalsvatn': 'Water_Level_Musdalsvatn',
                'Vanntemp. Musdalsvatn nedstrøms': 'Water_Temperature_Musdalsvatn_Downstream',
                'Vannstand Musdalsvatn nedstrøms': 'Water_Level_Musdalsvatn_Downstream',
                'Vanntemp. Viglesdalsvatn': 'Water_Temperature_Viglesdalsvatn',
                'Vannstand Viglesdalsvatn': 'Water_Level_Viglesdalsvatn',
                'Q_HBV': 'Flow_HBV',
                'PRECIP_HBV': 'Precipitation_HBV',
                'TEMP_HBV': 'Temperature_HBV',
                'SNOW_MELT_HBV': 'SNOW_MELT_HBV',
                'SNOW_SWE_HBV': 'Snow_Water_Equivalent_HBV',
                'Evap_HBV': 'Evaporation_HBV',
                'SOIL_WAT_HBV': 'Soil_Water_Storage_HBV',
                'GR_WAT_HBV': 'Groundwater_Storage_HBV',
                'Q_Kalltveit_uten_tapping': 'Flow_Without_Tapping_Kalltveit',
                'Q_HBV_mean': 'Mean_Flow_HBV',
                'Q_Lyngsaana': 'Flow_Lyngsaana',
                'Vanntemp. Lyngsaana': 'Water_Temperature_Lyngsaana',
                'Vanntemp. Kalltveit elv': 'Water_Temperature_Kalltveit_River',
                'Vannstand Lyngsåna': 'Water_Level_Lyngsaana',
                'Vanntemp. Lyngsåna': 'Water_Temperature_Lyngsaana'
            })
            


            return data
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
                    

    def create_lagged_matrix(self, window_size, vars_to_lag=None): #TODO: Fix decimal (five values in dataframe four in X, y)
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

        # Create variable name abbreviations
        var_abbrevs = {var: var[0] for var in vars_to_lag} if vars_to_lag else {}

        # Create abbreviated file name
        var_str = "_".join([var_abbrevs[var] for var in vars_to_lag]) if vars_to_lag else ""
        file_name = f"{window_size}_lag_{var_str}.csv"

        # Construct file path using os.path.join
        path = os.path.join(
            self.data_dir, 
            "clean_data", 
            "multivariate", 
            self.target_variable.replace(" ", "_"), 
            file_name)
            
        if os.path.isfile(path):
            lagged_df = self.load_data_from_file(path)
        else:
            data = self.get_raw_data()

            if vars_to_lag:
                lagged_df = data[[self.target_variable] + vars_to_lag].copy()
                # create a lagged matrix of target and variables
                for var in [self.target_variable] + vars_to_lag:
                    for i in range(1, window_size+1):
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
        if vars_to_lag:
            X = lagged_df.drop(columns=[self.target_variable] + vars_to_lag, axis=1)
        else:
            X = lagged_df.drop(columns=[self.target_variable], axis=1)
        y = lagged_df[f'{self.target_variable}']

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
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_size, shuffle=True)

        # Compute the validation size relative to the remaining data after the train split
        val_size_ratio = test_size / (val_size + test_size)
        
        # Split the remaining data into training and validation sets
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test,
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