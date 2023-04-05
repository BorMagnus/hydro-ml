from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

import os
import sys
from itertools import chain, combinations
import random

import plotly.graph_objects as go
import numpy as np

from train import train_model


def main(exp_name, n_samples, max_num_epochs, min_num_epochs, local_dir="ray_results"):
    target_variable = 'Flow_Kalltveit'
    file_name = "cleaned_data_1.csv"
    datetime_variable = "Datetime"

    models = [
                "LSTM", "LSTMTemporalAttention", "LSTMSpatialTemporalAttention"
            ] # Can be: "FCN", "FCNTemporalAttention", "LSTMTemporalAttention", "LSTM", "LSTMSpatialAttention", "LSTMSpatialTemporalAttention"

    def random_combinations(variables):
        num_combinations = 10
        min_variables = 1
        max_variables = len(variables)
        return [random.sample(variables, random.randint(min_variables, max_variables)) for _ in range(num_combinations)]

    variable_list =[ #TODO: get columns from file.
                "Wind_Speed_Nilsebu",
                "Air_Temperature_Nilsebu",
                "Wind_Direction_Nilsebu",
                "Relative_Humidity_Nilsebu",
                "Air_Temperature_Fister",
                "Precipitation_Fister",
                "Flow_Lyngsvatn_Overflow",
                "Flow_Tapping",
                "Water_Level_Kalltveit",
                "Water_Temperature_Kalltveit_Kum",
                "Precipitation_Nilsebu",
                "Flow_HBV",
                "Precipitation_HBV",
                "Temperature_HBV",
                "Flow_Without_Tapping_Kalltveit",
                "Flow_Lyngsaana",
                "Water_Temperature_Lyngsaana"
                ]
    config = {
        "data_file": file_name,
        "datetime":  datetime_variable,
        
        "data": {
            "target_variable": target_variable,
            "sequence_length": tune.choice([25]),
            "batch_size": tune.choice([256, 512]),
            "variables": tune.grid_search([]+random_combinations(variable_list))
        },

        "model": tune.grid_search(models), 
        "model_arch": {
            "input_size": None,
            "hidden_size": tune.choice([32, 64]),
            'num_layers': tune.choice([2, 3, 4]),
            "output_size": 1
        },

        "training": {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.choice([0, 0.001, 0.0001]),
        },

        'num_epochs': tune.choice([max_num_epochs]),
    }

    reporter = tune.CLIReporter(
        metric_columns=[
            "train_loss", "val_loss", "test_loss", "training_iteration"
        ])

    scheduler_asha = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=min_num_epochs,
        reduction_factor=2)

    scheduler_population = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=min_num_epochs,
        hyperparam_mutations={
            "training": {
                "learning_rate": lambda: 10 ** np.random.uniform(-4, -1),
                "weight_decay": [0, 0.001, 0.0001],
            },
        },
    )

    stop = {
        "training_iteration": max_num_epochs,
    }

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    results = tune.run(
        train_model, # TODO: partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        num_samples=n_samples,
        scheduler=scheduler_population,
        progress_reporter=reporter,
        name=exp_name,
        local_dir=local_dir,
        metric='val_loss',
        mode='min',
        stop=stop,
    )

if __name__=="__main__":
    main(exp_name="data_1-inflow_forecasting", n_samples=1, max_num_epochs=200, min_num_epochs=50)