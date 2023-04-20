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


def get_variables_combinations(file_name, datetime_variable):

    d = Data(file_name, datetime_variable)
    variables = d.get_all_variables()
    variables.remove("Flow_Kalltveit")

    univariate = []
    nilsebu = []
    lyngsaana = []
    hiafossen = []
    fister = []
    kalltveit = []
    hiavatn = []
    musdalsvatn = []
    viglesdalsvatn = []
    discharge = []
    meteorological = []
    hydrological = []
    hbv = []
    
    for var in variables:
        if "Nilsebu" in var:
            nilsebu.append(var)
        if "Lyngsaana" in var:
            lyngsaana.append(var)
        if "Hiafossen" in var:
            hiafossen.append(var)
        if "Fister" in var:
            fister.append(var)
        if "Kalltveit" in var:
            kalltveit.append(var)
        if "Hiavatn" in var:
            hiavatn.append(var)
        if "Musdalsvatn" in var:
            musdalsvatn.append(var)
        if "Viglesdalsvatn" in var:
            viglesdalsvatn.append(var)
        if "HBV" in var:
            hbv.append(var)
        if "Precipitation" in var or "Wind_Speed" in var or "Wind_Direction" in var or "Relative_Humidity" in var:
            meteorological.append(var)
        if "Water" in var:
            hydrological.append(var)
        if "Flow" in var:
            discharge.append(var)

        # Remove HBV values from the meteorological, hydrological, and discharge lists
        meteorological = [var for var in meteorological if "HBV" not in var]
        hydrological = [var for var in hydrological if "HBV" not in var]
        discharge = [var for var in discharge if "HBV" not in var]

    all_variables_combinations = [
        univariate,
        nilsebu,
        fister,
#        kalltveit,
#        lyngsaana,
        #nilsebu+fister,
        #hbv,
        meteorological,
        hydrological,
        #meteorological + hbv,
        #hydrological + hbv,
        meteorological + hydrological,
        #meteorological + hydrological + hbv,
    ]
    
    return all_variables_combinations


def main(
    model,
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
        #"LSTM",
        #"LSTMTemporalAttention",
        #"LSTMSpatialAttention",
        "LSTMSpatialTemporalAttention",
    ]  # Can be: "FCN", "FCNTemporalAttention", "LSTMTemporalAttention", "LSTM", "LSTMSpatialAttention", "LSTMSpatialTemporalAttention"

    config = {
        "data_file": file_name,
        "datetime": datetime_variable,
        "data": { 
            "target_variable": target_variable,
            "sequence_length": tune.choice([12, 25, 50]),
            "batch_size": tune.choice([64, 128, 256]),
            "variables": tune.grid_search(get_variables_combinations(file_name, datetime_variable)),
            "split_size": {"train_size": 0.7, "val_size": 0.2, "test_size": 0.1},
        },
        "model": tune.grid_search(model),
        "model_arch": {
            "input_size": tune.sample_from(
                lambda spec: len(spec.config.data["variables"]) + 1
            ),
            "hidden_size": tune.choice([16, 32, 64, 128]),
            "num_layers": tune.choice([1, 2, 3, 4]),
            "output_size": 1,
        },
        "training": {
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
        },
        "num_epochs": max_num_epochs
    }

    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "val_loss", "test_loss", "training_iteration"]
    )

    scheduler_asha = ASHAScheduler(
        max_t=max_num_epochs, grace_period=min_num_epochs, reduction_factor=3
    )

    scheduler_population = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "model_arch.hidden_size": tune.choice([16, 32, 64, 128]),
            "model_arch.num_layers": tune.choice([1, 2, 3, 4]),
            "data.batch_size": tune.choice([64, 128, 256]),
        },
    )

    stop = {
        "training_iteration": max_num_epochs,
    }

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    results = tune.run(
        train_model,
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        num_samples=n_samples,
        #scheduler=scheduler_population,
        progress_reporter=reporter,
        name=exp_name,
        local_dir=local_dir,
        metric="val_loss",
        mode="min",
        stop=stop,
        #search_alg=...
        keep_checkpoints_num=1, 
        checkpoint_score_attr="val_loss"
    )


if __name__ == "__main__":
    data_dir = "./data"
    clean_data_dir = os.path.abspath(os.path.join(data_dir, "clean_data"))
    models = [
        "LSTM",
        "LSTMTemporalAttention",
        "LSTMSpatialAttention",
        "LSTMSpatialTemporalAttention",
    ]
    exp_names = ["lstm", "temporal", "spatial", "spatio_temporal"]
    for i in range(len(models)):
        # Loop through each datafile in the data directory
        for filename in os.listdir(clean_data_dir):
            # Get the full path of the file
            file_path = os.path.join(clean_data_dir, filename)

            num = filename.split("_")[2].split(".")[0]
            exp_name = exp_names[i]
            experiment = f"data_{num}-{exp_name}"

            main(
                [models[i]],
                exp_name=experiment,
                file_name=filename,
                n_samples=2,
                max_num_epochs=100,
                min_num_epochs=50,
            )

    