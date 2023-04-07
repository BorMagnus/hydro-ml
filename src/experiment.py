from functools import partial
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

import os
import random

import plotly.graph_objects as go
import numpy as np

from train import train_model
from data import Data


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
        "LSTM",
        "LSTMTemporalAttention",
        "LSTMSpatialTemporalAttention",
    ]  # Can be: "FCN", "FCNTemporalAttention", "LSTMTemporalAttention", "LSTM", "LSTMSpatialAttention", "LSTMSpatialTemporalAttention"

    def random_combinations(variables):
        num_combinations = 5
        min_variables = 0
        max_variables = len(variables)
        return [
            random.sample(variables, random.randint(min_variables, max_variables))
            for _ in range(num_combinations)
        ]

    d = Data(file_name, datetime_variable)
    variable_list = d.get_all_variables()
    variable_list.remove(target_variable)

    all_variables_combinations = random_combinations(variable_list)

    config = {
        "data_file": file_name,
        "datetime": datetime_variable,
        "data": {
            "target_variable": target_variable,
            "sequence_length": tune.choice([25]),
            "batch_size": tune.choice([256, 512]),
            "variables": tune.grid_search(all_variables_combinations),
            "split_size": {"train_size": 0.7, "val_size": 0.2, "test_size": 0.1},
        },
        "model": tune.grid_search(models),
        "model_arch": {
            "input_size": tune.sample_from(
                lambda spec: len(spec.config.data["variables"]) + 1
            ),
            "hidden_size": tune.choice([32, 64]),
            "num_layers": tune.choice([2, 3, 4]),
            "output_size": 1,
        },
        "training": {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.choice([0, 0.001, 0.0001]),
        },
        "num_epochs": tune.choice([max_num_epochs]),
    }

    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "val_loss", "test_loss", "training_iteration"]
    )

    scheduler_asha = ASHAScheduler(
        max_t=max_num_epochs, grace_period=min_num_epochs, reduction_factor=2
    )

    scheduler_population = PopulationBasedTraining(
        time_attr="training_iteration",
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
        partial(train_model),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        num_samples=n_samples,
        scheduler=scheduler_asha,
        progress_reporter=reporter,
        name=exp_name,
        local_dir=local_dir,
        metric="val_loss",
        mode="min",
        stop=stop,
    )


if __name__ == "__main__":
    data_dir = "./data"
    clean_data_dir = os.path.abspath(os.path.join(data_dir, "clean_data"))

    # Loop through each datafile in the data directory
    for filename in os.listdir(clean_data_dir):
        # Get the full path of the file
        file_path = os.path.join(clean_data_dir, filename)

        num = filename.split("_")[2].split(".")[0]

        exp_name = "test"
        experiment = f"data_{num}-{exp_name}"

        main(
            exp_name=experiment,
            file_name=filename,
            n_samples=1,
            max_num_epochs=200,
            min_num_epochs=50,
        )

        break
