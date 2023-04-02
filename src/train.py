import torch
from ray import tune
from tqdm import tqdm
from ray.air import session
import os
import torch.optim as optim
import torch.nn as nn

from src import models as m
from src import data as d


def fit(net, loss_function, optimizer, data_loader, num_epochs, mode, checkpoint_dir, use_amp=False):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # Mixed-precision support for compatible GPUs
    losses = {"train": [], "val": [], "test": []}  # Add this line to store losses
    for epoch in range(num_epochs):
        if epoch < num_epochs - 1:
            keys = ["train", "val"]
        else:
            keys = ["train", "val", "test"]
        epoch_losses = {"train": 0.0, "val": 0.0, "test": 0.0}
        for key in keys:
            dataset_size = 0
            dataset_loss = 0.0
            if key == "train":
                net.train()
            else:
                net.eval()
            for X_batch, y_batch in tqdm(data_loader[key]):
                X_batch, y_batch = X_batch.to(mode["device"]), y_batch.to(mode["device"])
                with torch.set_grad_enabled(mode=(key=="train")): # Autograd activated only during training
                    with torch.cuda.amp.autocast(enabled=False): # Mixed-precision support for compatible GPUs
                        batch_output = net(X_batch.float())
                        batch_loss = loss_function(batch_output, y_batch)
                    if key == "train":
                        scaler.scale(batch_loss).backward() # type: ignore
                        scaler.step(optimizer) 	
                        scaler.update()
                        optimizer.zero_grad()
                dataset_size += y_batch.shape[0]
                dataset_loss += y_batch.shape[0] * batch_loss.item()

            dataset_loss /= dataset_size
            epoch_losses[key] = dataset_loss
            losses[key].append(dataset_loss)

        # Report results to Ray Tune after processing all keys
        tune.report(train_loss=epoch_losses["train"], val_loss=epoch_losses["val"], test_loss=epoch_losses["test"])

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
    return net


def train_model(config, checkpoint_dir=None):
    use_GPU = torch.cuda.is_available()
    if use_GPU:
        mode = {"name": "cuda", "device": torch.device("cuda")}
    else:
        mode = {"name": "cpu", "device": torch.device("cpu")}

    # Define hyperparameters
    train_size = 0.7
    val_size = 0.2
    test_size = 0.1

    sequence_length = config['sequence_length']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['learning_rate']
    weight_decay = config['weigth_decay']

    # Set data file
    data_file = config['data_file']
    datetime_variable = config['datetime']

    data = d.Data(data_file, datetime_variable)

    # Select variables to use
    vars = config['variables']
    target_variable = config['target_variable']
    X, y = data.data_transformation(sequence_length=sequence_length, target_variable=target_variable, columns_to_transformation=vars)

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = data.split_data(X, y, train_size=train_size, val_size=val_size, test_size=test_size)
    train_dataloader = data.create_dataloader(X_train, y_train, sequence_length, batch_size=batch_size, shuffle=True)
    val_dataloader = data.create_dataloader(X_val, y_val, sequence_length, batch_size=batch_size, shuffle=False)
    test_dataloader = data.create_dataloader(X_test, y_test, sequence_length, batch_size=batch_size, shuffle=False)

    # Model inputs
    if vars:
        input_size = len(vars) + 1
    else:
        input_size = 1
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    output_size = 1

    if config['arch'] == "FCN":
        net = m.FCN(input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    )
    elif config['arch'] ==  "FCNTemporalAttention":
        net = m.FCNTemporalAttention(input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    )
    elif config['arch'] == "LSTM":
        net = m.LSTM(input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    )
    elif config['arch'] == "LSTMTemporalAttention":
        net = m.LSTMTemporalAttention(input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    )
    elif config['arch'] == "LSTMSpatialAttention":
        net = m.LSTMSpatialAttention(input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    )
    elif config['arch'] == "LSTMSpatialTemporalAttention":
        net = m.LSTMSpatialTemporalAttention(input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    )

    data_loader = {
    "train": train_dataloader,
    "val": val_dataloader,
    "test": test_dataloader,
    }
    
    net.to(mode["device"])

    loss_function = nn.MSELoss().to(mode["device"])
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    fit(net, loss_function, optimizer, data_loader, num_epochs, mode, checkpoint_dir, use_amp=True)