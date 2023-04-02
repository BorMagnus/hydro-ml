import streamlit as st
from ray import tune

import os
import sys


dataframe = st.session_state['df']
file_name = st.session_state['file_name']

datetime_variable = st.selectbox(
    'Select date variable:',
    list(dataframe.columns))

target_variable = st.selectbox(
    'Select target variable:',
    list(dataframe.columns))

# get a list of column names excluding datatime
exclude_columns = [datetime_variable, target_variable]
column_names = [col for col in dataframe.columns if col not in exclude_columns]

variables = st.multiselect(
    'Select variables to be used in forecast:',
    column_names,
    column_names)

config = {
    "data_file": file_name,
    "datetime":  datetime_variable,
    "target_variable": target_variable,
    "arch": tune.grid_search(["LSTM", "LSTMTemporalAttention", "LSTMSpatialTemporalAttention"]), # "FCN", "FCNTemporalAttention", "LSTMTemporalAttention", "LSTM", "LSTMSpatialAttention", "LSTMSpatialTemporalAttention"
    "sequence_length": tune.choice([25]),
    'num_epochs': tune.choice([30]),
    'num_layers': tune.choice([2, 3, 4]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "weigth_decay": tune.choice([0, 0.001, 0.0001]),
    "batch_size": tune.choice([256, 512]),
    "hidden_size": tune.choice([32, 64]),
    "variables": tune.grid_search([
        variables
    ])
}

reporter = tune.JupyterNotebookReporter(
        parameter_columns={
            "weigth_decay": "w_decay",
            "learning_rate": "lr",
            "num_epochs": "num_epochs"
        },
        metric_columns=[
            "train_loss", "val_loss", "test_loss", "training_iteration"
        ])
#TODO: Need to fix import problems
"""analysis = tune.run(
    train_model, # TODO: partial(train_cifar, data_dir=data_dir),
    resources_per_trial={"cpu": 12, "gpu": 1},
    config=config,
    num_samples=1,
    #scheduler=scheduler,
    progress_reporter=reporter,
    name="inflow_forecasting",
    
)"""