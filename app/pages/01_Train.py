import os
import random
import sys
from functools import partial
from typing import List

import pandas as pd
import ray
import streamlit as st
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from src.data import Data
from src.train import train_model

def get_datetime_and_target_variables(dataframe):
    """Get datetime and target variables from dataframe."""
    datetime_variable = st.selectbox("Select date variable:", list(dataframe.columns))

    target_variable_options = list(dataframe.columns)
    target_variable_options.remove(datetime_variable)

    target_variable = st.selectbox("Select target variable:", target_variable_options, index=9)

    return datetime_variable, target_variable


def get_variables_form(dataframe, datetime_variable, target_variable):
    """Get variables to be used in forecast."""
    variables_options = list(dataframe.columns)
    variables_options.remove(datetime_variable)
    variables_options.remove(target_variable)

    variables_set = []
    with st.form(key="grid_search_form"):
        num_var_sets = st.number_input(
            "How many feature sets do you want to create?", min_value=1, value=1
        )

        for i in range(num_var_sets):
            variables = st.multiselect(
                f"Select variables for feature set {i + 1}:",
                variables_options,
                variables_options,
                key=f"feature_set_{i}",
            )
            variables_set.append(variables)

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        st.session_state["form_submitted"] = True
        st.session_state["variables_set"] = variables_set
    elif "form_submitted" not in st.session_state:
        st.session_state["form_submitted"] = False
        st.session_state["variables_set"] = [variables_options]
    return st.session_state["variables_set"]


def get_training_parameters_form():
    with st.form(key="training_parameters_form"):
        sequence_length = st.number_input(
            "Sequence length:", min_value=1, value=25, max_value=72
        )
        batch_size = st.number_input(
            "Batch size:", min_value=1, value=256, max_value=512
        )
        hidden_size = st.number_input(
            "Hidden size:", min_value=1, value=32, max_value=64
        )
        num_layers = st.number_input(
            "Number of layers:", min_value=1, value=2, max_value=4
        )
        learning_rate = st.number_input(
            "Learning rate:", min_value=0.0, value=1e-4, step=1e-4, format="%f"
        )
        weight_decay = st.number_input(
            "Weight decay:", min_value=0.0, value=0.0, step=1e-4, format="%f"
        )
        num_epochs = st.number_input(
            "Number of epochs:", min_value=1, value=100, max_value=200
        )

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        st.session_state["training_parameters_submitted"] = True
        st.session_state["training_parameters"] = {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
        }
    elif "training_parameters_submitted" not in st.session_state:
        st.session_state["training_parameters_submitted"] = False
        st.session_state["training_parameters"] = {
            "sequence_length": tune.choice([25]),
            "batch_size": tune.choice([256, 512]),
            "hidden_size": tune.choice([32, 64]),
            "num_layers": tune.choice([2, 3, 4]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.choice([0, 0.001, 0.0001]),
            "num_epochs": tune.choice([50, 100, 150, 200]),
        }

    return st.session_state["training_parameters"]


def select_models() -> List[str]:
    """Select models to use."""
    models = []
    if st.checkbox("FCN"):
        models.append("FCN")
    if st.checkbox("LSTM"):
        models.append("LSTM")
    if st.checkbox("LSTMTemporalAttention"):
        models.append("LSTMTemporalAttention")
    if st.checkbox("LSTMSpatioTemporalAttention"):
        models.append("LSTMSpatioTemporalAttention")

    if not models:
        st.warning("Please select at least one model.")
        st.stop()

    return models


def create_config(
    file_name,
    datetime_variable,
    target_variable,
    models,
    variables,
    training_parameters,
):
    """Create configuration for training."""

    config = {
        "data_file": file_name,
        "datetime": datetime_variable,
        "data": {
            "target_variable": target_variable,
            "sequence_length": training_parameters["sequence_length"],
            "batch_size": training_parameters["batch_size"],
            "variables": tune.grid_search(variables),
            "split_size": {"train_size": 0.7, "val_size": 0.2, "test_size": 0.1},
        },
        "model": tune.grid_search(models),
        "model_arch": {
            "input_size": tune.sample_from(
                lambda spec: len(spec.config.data["variables"]) + 1
            ),
            "hidden_size": training_parameters["hidden_size"],
            "num_layers": training_parameters["num_layers"],
            "output_size": 1,
        },
        "training": {
            "learning_rate": training_parameters["learning_rate"],
            "weight_decay": training_parameters["weight_decay"],
        },
        "num_epochs": training_parameters["num_epochs"],
    }

    return config


def train(config, exp_name, n_samples, min_num_epochs, max_num_epochs):
    """Run the training with given configuration."""

    scheduler_asha = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=min_num_epochs,
        reduction_factor=2,
    )

    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "val_loss", "test_loss", "training_iteration"]
    )

    stop = {"training_iteration": max_num_epochs}

    local_dir = "../ray_results/"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    analysis = tune.run(
        partial(train_model),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        num_samples=n_samples,
        #scheduler=scheduler_asha,
        progress_reporter=reporter,
        name=exp_name,
        local_dir=local_dir,
        metric="val_loss",
        mode="min",
        stop=stop,
    )

    st.session_state["analysis"] = analysis


def layout():
    st.title("Training")

    if "df" and "file_name" in st.session_state:
        dataframe = st.session_state["df"]
        file_name = st.session_state["file_name"]
        datetime_variable, target_variable = get_datetime_and_target_variables(
            dataframe
        )

        st.write("Select Models to use/compare:")
        models = select_models()

        with st.expander("Select variables sets"):
            variables = get_variables_form(
                dataframe, datetime_variable, target_variable
            )
            st.write("Sets of variables:")
            st.write(variables)

        with st.expander("Select parameters for the model"):
            training_parameters = get_training_parameters_form()
            st.write("Parameters:")
            st.write(training_parameters)

        config = create_config(
            file_name,
            datetime_variable,
            target_variable,
            models,
            variables,
            training_parameters,
        )

        col1, col2, col3, col4 = st.columns(4)

        # Place widgets in the columns
        exp_name = col1.text_input("Current experiment name:", "experiment_1")
        n_samples = col2.number_input(
            "Samples to perform", key="n_samples", step=1, value=1, min_value=1
        )
        min_num_epochs = col3.number_input(
            "Min number of epochs", key="min_num_epochs", step=10, min_value=1, value=50
        )
        max_num_epochs = col4.number_input(
            "Max number of epochs",
            key="max_num_epochs",
            step=10,
            min_value=1,
            value=100,
        )

        training_button = st.button("Start Training")
        if training_button:
            with st.spinner("Training..."):
                train(config, exp_name, n_samples, min_num_epochs, max_num_epochs)
            st.success("Done!")

        if training_button and "analysis" in st.session_state:
            results = st.session_state["analysis"]
            st.write("Trained models")
            st.dataframe(
                results.dataframe()[
                    [
                        "config/model",
                        "train_loss",
                        "val_loss",
                        "time_total_s",
                        "config/data/variables",
                    ]
                ]
            )

    else:
        st.error("Need to upload file!")


if __name__ == "__main__":
    layout()
