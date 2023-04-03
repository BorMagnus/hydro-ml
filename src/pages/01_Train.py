import streamlit as st
from ray import tune

import os
import sys

# Get the absolute path of the grandparent directory (Hydro-ML)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add the root directory to sys.path
sys.path.append(root_dir)

from src.train import train_model


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

agree_1 = st.checkbox('FCN')
agree_2 = st.checkbox('FCNTemporalAttention')
agree_3 = st.checkbox('LSTM')
agree_4 = st.checkbox('LSTMTemporalAttention')
#agree_5 = st.checkbox('LSTMSpatialAttention')
agree_6 = st.checkbox('LSTMSpatialTemporalAttention')

config = {
    "data_file": file_name,
    "datetime":  datetime_variable,
    "target_variable": target_variable,
    "model": {
        "arch": tune.grid_search(["LSTM"]), # "FCN", "FCNTemporalAttention", "LSTMTemporalAttention", "LSTM", "LSTMSpatialAttention", "LSTMSpatialTemporalAttention"
        'num_layers': tune.choice([2, 3, 4]),
        "hidden_size": tune.choice([32, 64]),
    },
    "data": {
        "sequence_length": tune.choice([25]),
        "batch_size": tune.choice([256, 512]),
        "variables": tune.grid_search([
         variables
        ])
    },
    "training": {
        'num_epochs': tune.choice([30]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weigth_decay": tune.choice([0, 0.001, 0.0001]),
    }
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

def train():
    analysis = tune.run(
        train_model, # TODO: partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        num_samples=1,
        #scheduler=scheduler,
        progress_reporter=reporter,
        name="inflow_forecasting",
    )
    st.session_state['analysis'] = analysis


if st.button('Train'):
    with st.spinner('Wait for it...'):
        train()
    st.success('Done!')
    
