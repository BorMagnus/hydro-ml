import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from pmdarima.arima import auto_arima
from data import Data

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
    mape = mean_absolute_percentage_error(test, predictions)
    r2 = r2_score(test, predictions)

    return mae, mse, rmse, mape, r2

def historical_average(data, window=25, iterations=1):
    train_size = int(len(data) * 0.9)
    train, test = data[:train_size], data[train_size:]

    predictions = []
    for _ in range(iterations):
        history = train[-window:].tolist()
        pred = np.mean(history)
        predictions.append(pred)
        train = np.append(train, pred)

    mae = mean_absolute_error(test[:iterations], predictions)
    mse = mean_squared_error(test[:iterations], predictions)
    rmse = mean_squared_error(test[:iterations], predictions, squared=False)
    mape = mean_absolute_percentage_error(test[:iterations], predictions)
    r2 = r2_score(test[:iterations], predictions)

    return mae, mse, rmse, mape, r2

def prepare_data(file_path):
    d = Data(file_path, "Datetime")
    data = d.get_data()
    target_data = data["Flow_Kalltveit"].values
    return target_data

if __name__ == "__main__":
    file_names = [
        "cleaned_data_4.csv",
    ]

    for i, file_path in enumerate(file_names, 1):
        target_data = prepare_data(file_path)
        best_order = find_best_order(target_data)
        mae_arima, mse_arima, rmse_arima, mape_arima, r2_arima = train_arima(target_data, best_order)
        mae_ha, mse_ha, rmse_ha, mape_ha, r2_ha = historical_average(target_data)

        print(f"Dataset {i}:")
        print("ARIMA Model:")
        print(f"MAE: {mae_arima:.4f}")
        print(f"MSE: {mse_arima:.4f}")
        print(f"RMSE: {rmse_arima:.4f}")
        print(f"MAPE: {mape_arima:.4f}%")
        print(f"R2: {r2_arima:.4f}")
        print()
        print("HA Model:")
        print(f"MAE: {mae_ha:.4f}")
        print(f"MSE: {mse_ha:.4f}")
        print(f"RMSE: {rmse_ha:.4f}")
        print(f"MAPE: {mape_ha:.4f}%")
        print(f"R2: {r2_ha:.4f}")

        mae_ha, mse_ha, rmse_ha, mape_ha, r2_ha = historical_average(target_data, iterations=12)
        print("HA Model 12 step ahead:")
        print(f"MAE: {mae_ha:.4f}")
        print(f"MSE: {mse_ha:.4f}")
        print(f"RMSE: {rmse_ha:.4f}")
        print(f"MAPE: {mape_ha:.4f}%")
        print(f"R2: {r2_ha:.4f}")