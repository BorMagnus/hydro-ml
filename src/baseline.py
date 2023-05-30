import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pmdarima.arima import auto_arima
import time
import sys
from src.data import Data

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100


def find_best_order(data):
    best_fit = auto_arima(data, seasonal=False, suppress_warnings=True, stepwise=True)
    return best_fit.order

def historical_average(data, window=25):
    train_size = int(len(data) * 0.9)
    train, test = data[:train_size], data[train_size:]

    predictions = []
    forecast_history = train[-window:].tolist()
    start_time = time.time()  # Start measuring prediction time
    for _ in range(len(test)):
        pred = np.mean(forecast_history[-window:])
        predictions.append(pred)
        forecast_history.append(test[_])
    end_time = time.time()  # Stop measuring prediction time

    # Inverse the transformation
    predictions = np.cumsum(predictions)
    test = np.cumsum(test)

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = mean_squared_error(test, predictions, squared=False)
    mape = mean_absolute_percentage_error(test, predictions)
    r2 = r2_score(test, predictions)

    # get the size of the model object in memory
    model_size_bytes = sys.getsizeof(forecast_history)
    # convert to KB
    model_size_kb = model_size_bytes / 1024

    prediction_time = end_time - start_time

    return mae, mse, rmse, mape, r2, model_size_kb, prediction_time

def historical_average_multistep(data, window=25, steps=12):
    train_size = int(len(data) * 0.9)
    train, test = data[:train_size], data[train_size:]

    predictions = []
    forecast_history = train[-window:].tolist()
    i = 0
    for _ in range(0, len(test), steps):
        for _ in range(steps):
            if len(predictions)== len(test):
                break
            pred = np.mean(forecast_history[-window:])
            predictions.append(pred)
            forecast_history.append(pred)
            i+=1

    # Inverse the transformation
    predictions = np.cumsum(predictions)
    test = np.cumsum(test)

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = mean_squared_error(test, predictions, squared=False)
    mape = mean_absolute_percentage_error(test, predictions)
    r2 = r2_score(test, predictions)

    return mae, mse, rmse, mape, r2

def train_arima(target_data, order):
    start_time = time.time()
    
    # split the data into training and test sets
    train_size = int(len(target_data) * (1 - 0.1))
    train_data, test_data = target_data[:train_size], target_data[train_size:]
    
    # fit the ARIMA model to the training data
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    # generate predictions on the test data
    start_prediction_time = time.time()  # Start measuring prediction time
    predictions = model_fit.predict(start=train_size, end=len(target_data)-1, typ='levels')
    end_prediction_time = time.time()  # Stop measuring prediction time

    # Inverse the transformation
    predictions = np.cumsum(predictions)
    test_data = np.cumsum(test_data)

    end_time = time.time()
    training_time_arima = end_time - start_time
    prediction_time_arima = end_prediction_time - start_prediction_time
    mae_arima = mean_absolute_error(test_data, predictions)
    mse_arima = mean_squared_error(test_data, predictions)
    rmse_arima = np.sqrt(mse_arima)
    mape_arima = mean_absolute_percentage_error(test_data, predictions)
    r2_arima = r2_score(test_data, predictions)

    # get the size of the model object in memory
    model_size_bytes = sys.getsizeof(model)

    # convert to KB
    model_size_kb = model_size_bytes / 1024

    return mae_arima, mse_arima, rmse_arima, mape_arima, r2_arima, training_time_arima, prediction_time_arima, model_size_kb


def recursive_arima(target_data, order, forecast_horizon=1):
    
    # split the data into training and test sets
    train_size = int(len(target_data) * (1 - 0.1))
    train_data, test_data = target_data[:train_size], target_data[train_size:]
    
    # fit the ARIMA model to the training data
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    # make recursive predictions on the test data
    predictions = []
    for i in range(len(test_data)):
        # forecast the next value based on the previous forecast_horizon values
        if i < forecast_horizon:
            pred = model_fit.predict(start=i, end=i, typ='levels')
        else:
            input_data = predictions[i-forecast_horizon:i]
            pred = model_fit.predict(start=i, end=i, exog=input_data, typ='levels')
        predictions.append(pred[0])

    # Inverse the transformation
    predictions = np.cumsum(predictions)
    test_data = np.cumsum(test_data)
    
    mae_arima = mean_absolute_error(test_data, predictions)
    mse_arima = mean_squared_error(test_data, predictions)
    rmse_arima = np.sqrt(mse_arima)
    mape_arima = mean_absolute_percentage_error(test_data, predictions)
    r2_arima = r2_score(test_data, predictions)
    return mae_arima, mse_arima, rmse_arima, mape_arima, r2_arima


def prepare_data(file_path):
    d = Data(file_path, "Datetime")
    data = d.get_data()
    return data["Flow_Kalltveit"]

def make_data_stationary(data):
    df = pd.DataFrame(data)
    df.columns = ['target_data']

    df['stationary_target'] = df['target_data'].diff().dropna()

    # Check if the data is now stationary using Augmented Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df['stationary_target'].dropna())
    
    #print('ADF Statistic: %f' % result[0])
    #print('p-value: %f' % result[1])
    
    # The null hypothesis of the ADF test is that the time series is non-stationary. 
    # So, if the p-value of the test is less than the significance level (0.05) then 
    # you reject the null hypothesis and infer that the time series is indeed stationary.
    
    return df['stationary_target'].dropna()

if __name__ == "__main__":

    data = prepare_data("cleaned_data_4.csv")
    target_data = data.values

    stationary_data = make_data_stationary(target_data)
    stationary_target_data = stationary_data.values
    order = find_best_order(stationary_target_data)

    mae_arima, mse_arima, rmse_arima, mape_arima, r2_arima, training_time_arima, pred_time_arima, arima_size_kb = train_arima(stationary_target_data, order=order)
    mae_ha, mse_ha, rmse_ha, mape_ha, r2_ha, ha_size_kb, pred_time_HA = historical_average(stationary_target_data)
    print("ARIMA Model 1 hour ahead:")
    print(f"MAE: {mae_arima:.4f}")
    print(f"MSE: {mse_arima:.4f}")
    print(f"RMSE: {rmse_arima:.4f}")
    print(f"MAPE: {mape_arima:.4f}")
    print(f"R2: {r2_arima:.4f}")
    print(f"Training Time: {training_time_arima:.2f} seconds")
    print(f"Pred Time: {pred_time_arima}")
    print(f"Model size {arima_size_kb}KB")
    print()
    print("HA Model 1 hour ahead:")
    print(f"MAE: {mae_ha:.4f}")
    print(f"MSE: {mse_ha:.4f}")
    print(f"RMSE: {rmse_ha:.4f}")
    print(f"MAPE: {mape_ha:.4f}")
    print(f"R2: {r2_ha:.4f}")
    print(f"Pred Time: {pred_time_HA}")
    print(f"Model size {ha_size_kb}KB")
    print()
    
    mae_arima_12, mse_arima_12, rmse_arima_12, mape_arima_12, r2_arima_12 = recursive_arima(stationary_target_data, order, forecast_horizon=12)
    mae_ha_12, mse_ha_12, rmse_ha_12, mape_ha_12, r2_ha_12 = historical_average_multistep(stationary_target_data, steps=12)
    print("ARIMA Model 12 hour ahead:")
    print(f"MAE: {mae_arima_12:.4f}")
    print(f"MSE: {mse_arima_12:.4f}")
    print(f"RMSE: {rmse_arima_12:.4f}")
    print(f"MAPE: {mape_arima_12:.4f}")
    print(f"R2: {r2_arima_12:.4f}")
    print()
    print("HA Model 12 hour ahead:")
    print(f"MAE: {mae_ha_12:.4f}")
    print(f"MSE: {mse_ha_12:.4f}")
    print(f"RMSE: {rmse_ha_12:.4f}")
    print(f"MAPE: {mape_ha_12:.4f}")
    print(f"R2: {r2_ha_12:.4f}")
