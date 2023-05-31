import json
from pathlib import Path
from operator import itemgetter
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
import os

import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.express as px
from plotly.subplots import make_subplots

from src.data import *
from src.train import create_model


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    return (
        np.mean(
            np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])
        )
        * 100
    )


# Function to find best checkpoints for a model directory
def find_best_checkpoints(model_dir, num_best=5):
    checkpoints = []

    # Iterate over all training runs in the model directory
    for run_dir in model_dir.iterdir():
        if run_dir.is_dir():
            # Read the progress.csv file to get the validation losses
            progress_file = run_dir / "progress.csv"
            if progress_file.exists():
                with open(run_dir / "params.json", "r") as f:
                    params = json.load(f)
                progress_data = pd.read_csv(progress_file)

                best_val_idx = progress_data["val_loss"].idxmin()
                best_val_loss = progress_data.loc[best_val_idx, "val_loss"]

                # Save the checkpoint path and validation loss
                checkpoint_path = run_dir / "my_model" / "checkpoint.pt"
                checkpoints.append((checkpoint_path, best_val_loss, params))

    # Sort the checkpoints based on validation loss
    checkpoints.sort(key=itemgetter(1))

    return checkpoints[:num_best]


def calculate_model_metrics(model_dirs, experiment, best):
    model_dfs = {}
    parameters = []
    for model_dir in model_dirs:
        if experiment not in str(model_dir):
            continue
        rows = []
        best_checkpoints = find_best_checkpoints(model_dir, num_best=best)
        for i, (checkpoint, val_loss, params) in enumerate(best_checkpoints):
            # Load model and weights
            model = create_model(params)
            model = load_model_from_checkpoint(model, checkpoint)

            data_loader, scalers = get_dataloader(params)
            test_loader = data_loader["test"]
            with torch.no_grad():
                start_time = time.time()
                y_preds, y_test = get_preds_actuals(model, test_loader)
                end_time = time.time()

            d = Data(params["data_file"], params["datetime"])

            y_preds = d.inverse_transform_target(
                np.array(y_preds).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )
            y_test = d.inverse_transform_target(
                np.array(y_test).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )

            # Calculate the Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, y_preds)
            # Calculate the Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mean_squared_error(y_test, y_preds))
            # Calculate the Mean Absolute Percentage Error (MAPE)
            mape = mean_absolute_percentage_error(y_test, y_preds)
            # Calculate the Determination Coefficient (R^2)
            r2 = r2_score(y_test, y_preds)
            # Testing time
            test_time = end_time - start_time

            variables_category = categorize_features(params)

            rows.append(
                {
                    "model": params["model"],
                    "val_mae": val_loss,
                    "test_mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "r2": r2,
                    "variables": variables_category,
                    "testing (s)": test_time,
                }
            )
            # Create a dictionary with the model name and all hyperparameters
            parameters.append(
                {
                    "model_name": params["model"],
                    "val_mea": val_loss,
                    "target_variable": params["data"]["target_variable"],
                    "sequence_length": params["data"]["sequence_length"],
                    "batch_size": params["data"]["batch_size"],
                    "variables": variables_category,
                    "train_size": params["data"]["split_size"]["train_size"],
                    "val_size": params["data"]["split_size"]["val_size"],
                    "test_size": params["data"]["split_size"]["test_size"],
                    "input_size": params["model_arch"]["input_size"],
                    "hidden_size": params["model_arch"]["hidden_size"],
                    "num_layers": params["model_arch"]["num_layers"],
                    "output_size": params["model_arch"]["output_size"],
                    "learning_rate": params["training"]["learning_rate"],
                    "weight_decay": params["training"]["weight_decay"],
                    "num_epochs": params["num_epochs"],
                }
            )
        df = pd.DataFrame(rows)
        model_dfs[model_dir.name] = df
    return model_dfs, parameters


def get_dataloader(params):
    d = Data(params["data_file"], params["datetime"])
    data_loader, scalers = d.prepare_data(**params["data"])
    return data_loader, scalers


def load_model_from_checkpoint(checkpoint_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def get_preds_actuals(model, test_dataloader):
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            batch_output = model(X_batch.float())
            all_preds.extend(batch_output.squeeze().numpy())
            all_actuals.extend(y_batch.numpy())

    return all_preds, all_actuals


def load_model_from_checkpoint(model, checkpoint_path):
    try:
        model_state_dict, _ = torch.load(checkpoint_path)
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {checkpoint_path}")
        print(e)
    return model


def load_model_from_checkpoint(model, checkpoint_path):
    try:
        model_state_dict, _ = torch.load(checkpoint_path)
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {checkpoint_path}")
        print(e)
    return model


def get_losses(run_dir):
    if run_dir.is_dir():
        # Read the progress.csv file to get the validation losses
        progress_file = run_dir / "progress.csv"
        if progress_file.exists():
            return pd.read_csv(progress_file)[["train_loss", "val_loss"]]


def plot_losses(losses):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=losses["train_loss"], mode="lines", name="Train Loss"))
    fig.add_trace(
        go.Scatter(y=losses["val_loss"], mode="lines", name="Validation Loss")
    )

    fig.update_layout(
        title="Train and Validation Losses", xaxis_title="Epoch", yaxis_title="Loss"
    )

    fig.show()


def categorize_features(params):
    """Categorize features into meteorological, hydrological, and hbv categories or all."""
    d = Data(params["data_file"], params["datetime"])
    features = params["data"]["variables"]
    all_features = d.get_all_variables()

    # Check if the features are equal to all possible features
    if set(features) == set(all_features):
        return "all"

    hbv = []
    meteorological = []
    hydrological = []

    for var in features:
        if "HBV" in var:
            hbv.append(var)
        if (
            "Precipitation" in var
            or "Wind_Speed" in var
            or "Wind_Direction" in var
            or "Relative_Humidity" in var
        ):
            meteorological.append(var)
        if "Water" in var:
            hydrological.append(var)

    # Create a string representing the category of the variables
    categories = []
    if meteorological:
        categories.append("meteorological")
    if hydrological:
        categories.append("hydrological")
    if hbv:
        categories.append("hbv")

    return " + ".join(categories)


def box_plot(df):
    fig = px.box(
        df, x="model", y="test_mae", title="Box Plot of Test MAE for each model"
    )
    fig.show()


def descriptive_statistics(df):
    summary_statistics = df.groupby("model").describe()
    print(summary_statistics)


def median_iqr(df):
    # Calculate median for each model
    medians = df.groupby("model").median(numeric_only=True)

    # Calculate IQR for each model
    Q1 = df.groupby("model").quantile(0.25, numeric_only=True)
    Q3 = df.groupby("model").quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1

    print("Medians:\n", medians)
    print("\nIQR:\n", IQR)


import pandas as pd
import json
from operator import itemgetter


def count_models(model_dirs, experiment):
    model_count = {}

    for model_dir in model_dirs:
        if experiment not in str(model_dir):
            continue
        model_dir_path = Path(model_dir)
        # Iterate over all training runs in the model directory
        for run_dir in model_dir_path.iterdir():
            if run_dir.is_dir():
                # Check if the progress.csv file exists
                progress_file = run_dir / "progress.csv"
                if progress_file.exists():
                    with open(run_dir / "params.json", "r") as f:
                        params = json.load(f)
                        model = params["model"]

                        # Increment the count for the model type
                        if model in model_count:
                            model_count[model] += 1
                        else:
                            model_count[model] = 1

    # Convert the model_count dictionary to a pandas dataframe
    model_count_df = pd.DataFrame(
        list(model_count.items()), columns=["Model_Type", "Count"]
    )

    return model_count_df


def find_best_checkpoints_with_time(model_dir, num_best=5):
    checkpoints = []

    # Iterate over all training runs in the model directory
    for run_dir in model_dir.iterdir():
        if run_dir.is_dir():
            # Read the progress.csv file to get the validation losses and training time
            progress_file = run_dir / "progress.csv"
            if progress_file.exists():
                with open(run_dir / "params.json", "r") as f:
                    params = json.load(f)
                progress_data = pd.read_csv(progress_file)

                best_val_idx = progress_data["val_loss"].idxmin()
                best_val_loss = progress_data.loc[best_val_idx, "val_loss"]
                training_time = progress_data.loc[best_val_idx, "time_total_s"]

                # Save the checkpoint path, validation loss, and training time
                checkpoint_path = run_dir / "my_model" / "checkpoint.pt"
                checkpoints.append(
                    (checkpoint_path, best_val_loss, training_time, params)
                )

    # Sort the checkpoints based on validation loss
    checkpoints.sort(key=itemgetter(1))

    return checkpoints[:num_best]


from plotly.subplots import make_subplots


def visualize_attention(
    attention_weights_spatial, attention_weights_temporal, batch_idx, features
):
    # Extract attention weights for a specific batch element
    attention_matrix_spatial = (
        attention_weights_spatial[batch_idx].detach().cpu().numpy()
    )
    attention_matrix_temporal = (
        attention_weights_temporal[batch_idx].detach().cpu().numpy()
    )

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Spatial Attention Weights", "Temporal Attention Weights"),
    )

    # Add spatial attention heatmap to subplot
    fig.add_trace(
        go.Heatmap(
            z=attention_matrix_spatial,
            x=[f"f{i}" for i in range(1, attention_matrix_spatial.shape[1] + 1)],
            y=[f"t-{i}" for i in range(1, attention_matrix_spatial.shape[0] + 1)],
            colorscale="Viridis",
            name="Spatial Weights",
        ),
        row=1,
        col=1,
    )

    # Add temporal attention heatmap to subplot
    fig.add_trace(
        go.Heatmap(
            z=attention_matrix_temporal,
            y=[f"t-{i}" for i in range(1, attention_matrix_temporal.shape[0] + 1)],
            x=[f"f{i}" for i in range(1, attention_matrix_temporal.shape[1] + 1)],
            colorscale="Viridis",
            name="Temporal Weights",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        width=1800,
        height=800,
        xaxis_title="Features",
        yaxis_title="Input Time Step",
    )
    fig.show()


def plot_attention(params, spatial_weights=None, temporal_weights=None):
    features = [params["data"]["target_variable"]] + params["data"]["variables"]
    b = 20
    # If the attention weights are torch tensors, convert them to numpy arrays first
    if isinstance(spatial_weights, torch.Tensor) and isinstance(
        temporal_weights, torch.Tensor
    ):
        visualize_attention(
            spatial_weights, temporal_weights, batch_idx=b, features=features
        )


def recursive_forecast(model, input, forecast_steps=1, return_weights=False):
    predictions = []
    attention_weights = []

    # Find the index of the target feature in the input_size dimension
    target_feature_idx = 0

    for i in range(forecast_steps):
        if return_weights:
            out, alpha_list, beta_t = model(input, return_weights=True)
            if i + 1 in {1, 4, 8, 12}:
                attention_weights.append((alpha_list, beta_t))
        else:
            out = model(input)

        predictions.append(out)

        input[:, -1, target_feature_idx] = out.squeeze(-1)

    predictions = torch.stack(predictions, dim=1)

    if return_weights:
        return predictions, attention_weights
    else:
        return predictions


def get_multi_step_preds_actuals(
    model, test_loader, forecast_steps=3, return_weights=False
):
    y_preds = []
    y_test = []
    attention_weights_all = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the model to the GPU
    model.to(device)

    for input, target in test_loader:
        # Calculate the number of elements that can be reshaped
        num_elements_to_reshape = target.shape[0] // forecast_steps * forecast_steps

        # Slice the input tensor to match the reshaped target tensor
        input_sliced = input[::forecast_steps][
            : num_elements_to_reshape // forecast_steps
        ]

        input_sliced = input_sliced.to(device)
        if return_weights:
            predictions, attention_weights = recursive_forecast(
                model, input_sliced, forecast_steps, return_weights=True
            )
            attention_weights_all.append(attention_weights)
        else:
            predictions = recursive_forecast(model, input_sliced, forecast_steps)

        y_preds.append(predictions)

        # Slice the target tensor to keep only the elements that can be reshaped
        target_sliced = target[:num_elements_to_reshape]

        # Reshape the sliced target tensor to match the forecast_steps
        reshaped_target = target_sliced.view(-1, forecast_steps)
        y_test.append(reshaped_target)

    y_preds = torch.cat(y_preds).detach().cpu().numpy()
    y_test = torch.cat(y_test).detach().cpu().numpy()

    if return_weights:
        return y_preds, y_test, attention_weights_all
    else:
        return y_preds, y_test


def plot_pred_actual_multi_step(model_dirs, experiment, steps_ahead):
    fig = go.Figure()

    for model_dir in model_dirs:
        if experiment not in str(model_dir):
            continue

        best_checkpoints = find_best_checkpoints(model_dir, num_best=1)
        for checkpoint, _, params in best_checkpoints:
            model = create_model(params)
            model = load_model_from_checkpoint(model, checkpoint)

            data_loader, scalers = get_dataloader(params)
            test_loader = data_loader["test"]

            d = Data(params["data_file"], params["datetime"])

            if model_dir == model_dirs[0]:
                val_loader = data_loader["val"]
                y_val = [j for _, j in val_loader]
                y_val = torch.cat(y_val).detach().cpu().numpy()
                y_val = d.inverse_transform_target(
                    np.array(y_val).reshape(-1, 1), scalers["Flow_Kalltveit"]
                )
                datetime_val = val_loader.datetime_index
                val_df = pd.DataFrame(
                    {"datetime": datetime_val, "target": y_val.flatten()}
                )
                fig.add_trace(
                    go.Scatter(
                        x=val_df["datetime"],
                        y=val_df["target"],
                        mode="lines",
                        name="Validation",
                    )
                )

                y_test = [j for _, j in test_loader]
                y_test = torch.cat(y_test).detach().cpu().numpy()
                y_test = d.inverse_transform_target(
                    np.array(y_test).reshape(-1, 1), scalers["Flow_Kalltveit"]
                )
                datetime_test = test_loader.datetime_index
                test_df = pd.DataFrame(
                    {"datetime": datetime_test, "target": y_test.flatten()}
                )
                fig.add_trace(
                    go.Scatter(
                        x=test_df["datetime"],
                        y=test_df["target"],
                        mode="lines",
                        name="Test",
                    )
                )

            fig = plot_pred_actual(
                fig,
                data_loader,
                scalers,
                model,
                params["model"],
                forecast_steps=steps_ahead,
            )

    fig.update_layout(
        title=f"Time Series Data with Multiple Models Predictions",
        xaxis_title="Datetime",
        yaxis_title="Target Value",
    )
    fig.show()
    return fig


def plot_pred_actual(model_dirs, experiment):
    fig = go.Figure()
    for model_dir in model_dirs:
        if experiment not in str(model_dir):
            continue
        best_checkpoints = find_best_checkpoints(model_dir, num_best=1)
        for i, (checkpoint, val_loss, params) in enumerate(best_checkpoints):
            # Load model and weights
            model = create_model(params)
            model = load_model_from_checkpoint(model, checkpoint)

            data_loader, scalers = get_dataloader(params)
            test_loader = data_loader["test"]

            val_loader, test_loader = data_loader["val"], data_loader["test"]

            y_val = [j for _, j in val_loader]
            y_val = torch.cat(y_val).detach().cpu().numpy()

            model_name = params["model"]

            with torch.no_grad():
                y_preds, y_test = get_preds_actuals(model, test_loader)

            d = Data(params["data_file"], params["datetime"])

            y_val = d.inverse_transform_target(
                np.array(y_val).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )
            y_preds = d.inverse_transform_target(
                np.array(y_preds).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )
            y_test = d.inverse_transform_target(
                np.array(y_test).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )

            # Get datetime values
            datetime_val = val_loader.datetime_index
            datetime_test = test_loader.datetime_index

            # Slice the datetime values to match the reshaped target values
            datetime_test_sliced = datetime_test[: len(y_preds)]

            # Create dataframes
            val_df = pd.DataFrame({"datetime": datetime_val, "target": y_val.flatten()})
            test_df = pd.DataFrame(
                {"datetime": datetime_test_sliced, "target": y_test.flatten()}
            )
            predictions_df = pd.DataFrame(
                {"datetime": datetime_test_sliced, "predictions": y_preds.flatten()}
            )

    fig.add_trace(
        go.Scatter(
            x=val_df["datetime"], y=val_df["target"], mode="lines", name="Validation"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_df["datetime"], y=test_df["target"], mode="lines", name="Test"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions_df["datetime"],
            y=predictions_df["predictions"],
            mode="lines",
            name="Predictions",
        )
    )

    # Add spread plot between Test and Predictions
    fig.add_trace(
        go.Scatter(
            x=test_df["datetime"],
            y=test_df["target"],
            fill=None,
            mode="lines",
            line_color="rgba(0, 0, 0, 0.1)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions_df["datetime"],
            y=predictions_df["predictions"],
            fill="tonexty",
            mode="lines",
            line_color="rgba(0, 0, 0, 0.1)",
            name="Spread",
        )
    )

    # Add axis labels and plot title
    fig.update_layout(
        title=f"Time Series Data with {model_name} Predictions",
        xaxis_title="Datetime",
        yaxis_title="Target Value",
    )

    # Show the plot
    return fig


def evaluate_multi_step_models(model_dirs, experiment, steps_ahead, best):
    model_dfs = {}
    for model_dir in model_dirs:
        if experiment not in str(model_dir):
            continue

        rows = []
        best_checkpoints = find_best_checkpoints(model_dir, num_best=best)
        for i, (checkpoint, val_loss, params) in enumerate(best_checkpoints):
            # Load model and weights
            model = create_model(params)
            model = load_model_from_checkpoint(model, checkpoint)

            data_loader, scalers = get_dataloader(params)
            test_loader = data_loader["test"]

            with torch.no_grad():
                y_preds, y_test = get_multi_step_preds_actuals(
                    model, test_loader, forecast_steps=steps_ahead
                )

            d = Data(params["data_file"], params["datetime"])

            y_preds = d.inverse_transform_target(
                np.array(y_preds).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )
            y_test = d.inverse_transform_target(
                np.array(y_test).reshape(-1, 1), scalers["Flow_Kalltveit"]
            )

            # Calculate the Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, y_preds)
            # Calculate the Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mean_squared_error(y_test, y_preds))
            # Calculate the Mean Absolute Percentage Error (MAPE)
            mape = mean_absolute_percentage_error(y_test, y_preds)
            # Calculate the Determination Coefficient (R^2)
            r2 = r2_score(y_test, y_preds)

            model_variables = params["data"]["variables"]
            variables_category = categorize_features(model_variables)

            rows.append(
                {
                    "model": params["model"],
                    "val_mae": val_loss,
                    "test_mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "r2": r2,
                    "variables": variables_category,
                }
            )
        df = pd.DataFrame(rows)
        model_dfs[model_dir.name] = df

    return model_dfs
