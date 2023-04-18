import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from pmdarima.arima import auto_arima
from data import Data

def find_best_order(data):
    best_fit = auto_arima(data, seasonal=False, suppress_warnings=True, stepwise=True)
    return best_fit.order

def train_arima(data, order):
    train_size = int(len(data) * 0.9)
    train, test = data[:train_size], data[train_size:]

    # Fit ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = mean_squared_error(test, predictions, squared=False)

    return mae, mse, rmse

def prepare_data(file_path):
    d = Data(file_path, "Datetime")
    data = d.get_data()
    target_data = data["Flow_Kalltveit"].values
    return target_data

if __name__ == "__main__":
    file_names = [
        "cleaned_data_1.csv",
        "cleaned_data_2.csv",
        "cleaned_data_3.csv",
        "cleaned_data_4.csv",
    ]

    for i, file_path in enumerate(file_names, 1):
        target_data = prepare_data(file_path)
        best_order = find_best_order(target_data)
        mae, mse, rmse = train_arima(target_data, best_order)
        
        print(f"Dataset {i}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}\n")
        print()