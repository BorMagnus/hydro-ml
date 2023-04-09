import os
import random
from functools import partial

import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.ax import AxSearch


from data import Data
from train import train_model


def get_variables_data(file_name):
    
    Univariate_variable = []

    Nilsebu_variables = [
        "Wind_Speed_Nilsebu",
        "Air_Temperature_Nilsebu",
        "Wind_Direction_Nilsebu",
        "Relative_Humidity_Nilsebu",
        "Precipitation_Nilsebu",
    ]

    Fister_variables = [
        "Air_Temperature_Fister",
        "Precipitation_Fister",
    ]

    Kalltveit_variables = [
        "Water_Level_Kalltveit",
        "Water_Temperature_Kalltveit_Kum",
        "Flow_Without_Tapping_Kalltveit",
    ]

    Lyngsaana_variables = [
        "Flow_Lyngsvatn_Overflow",
        "Flow_Tapping",
        "Flow_Lyngsaana",
        "Water_Temperature_Lyngsaana",
    ]

    HBV_variables = [
        "Flow_HBV",
        "Precipitation_HBV",
        "Temperature_HBV",
    ]

    meteorological_variables = [
        "Wind_Speed_Nilsebu",
        "Air_Temperature_Nilsebu",
        "Wind_Direction_Nilsebu",
        "Relative_Humidity_Nilsebu",
        "Air_Temperature_Fister",
        "Precipitation_Fister",
        "Precipitation_Nilsebu",
    ]

    hydrological_variables = [
        "Water_Level_Kalltveit",
        "Water_Temperature_Kalltveit_Kum",
        "Flow_Lyngsvatn_Overflow",
        "Flow_Tapping",
        "Flow_Without_Tapping_Kalltveit",
        "Flow_Lyngsaana",
        "Water_Temperature_Lyngsaana",
    ]

    all_variables_combinations = [
        Univariate_variable,
        Nilsebu_variables,
        Fister_variables,
        #Kalltveit_variables,
        #Lyngsaana_variables,
        Nilsebu_variables+Fister_variables,
        #HBV_variables,
        meteorological_variables,
        hydrological_variables,
        #meteorological_variables + HBV_variables,
        #hydrological_variables + HBV_variables,
        meteorological_variables + hydrological_variables,
        #meteorological_variables + hydrological_variables + HBV_variables,
    ]

    return all_variables_combinations


def main(
    exp_name,
    file_name,
    n_samples,
    max_num_epochs,
    min_num_epochs,
    local_dir="ray_results",
):
    target_variable = "Flow_Kalltveit"
    datetime_variable = "Datetime"

    models = [
        "LSTMSpatialTemporalAttention",
        "LSTMTemporalAttention",
        "LSTM",
    ]  # Can be: "FCN", "FCNTemporalAttention", "LSTMTemporalAttention", "LSTM", "LSTMSpatialAttention", "LSTMSpatialTemporalAttention"

    config = {
        "data_file": file_name,
        "datetime": datetime_variable,
        "data": {
            "target_variable": target_variable,
            "sequence_length": tune.choice([25]),
            "batch_size": tune.choice([128, 256, 512]),
            "variables": tune.grid_search(get_variables_data(file_name)),
            "split_size": {"train_size": 0.7, "val_size": 0.2, "test_size": 0.1},
        },
        "model": tune.grid_search(models),
        "model_arch": {
            "input_size": tune.sample_from(
                lambda spec: len(spec.config.data["variables"]) + 1
            ),
            "hidden_size": tune.choice([32, 64, 128]),
            "num_layers": tune.choice([1, 2, 3]),
            "output_size": 1,
        },
        "training": {
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
        },
        "num_epochs": 200 #tune.randint(100, 500),
    }

    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "val_loss", "test_loss", "training_iteration"]
    )

    scheduler_asha = ASHAScheduler(
        max_t=max_num_epochs, grace_period=min_num_epochs, reduction_factor=2
    )

    stop = {
        "training_iteration": max_num_epochs,
    }

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    results = tune.run(
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
        #search_alg=...
    )


if __name__ == "__main__":
    data_dir = "./data"
    clean_data_dir = os.path.abspath(os.path.join(data_dir, "clean_data"))

    # Loop through each datafile in the data directory
    for filename in os.listdir(clean_data_dir):
        # Get the full path of the file
        file_path = os.path.join(clean_data_dir, filename)

        num = filename.split("_")[2].split(".")[0]

        exp_name = "location_based"
        experiment = f"data_{num}-{exp_name}"

        main(
            exp_name=experiment,
            file_name=filename,
            n_samples=1,
            max_num_epochs=500,
            min_num_epochs=100,
        )
        break

 