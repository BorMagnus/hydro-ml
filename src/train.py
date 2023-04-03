import torch
from ray import tune
from tqdm import tqdm
from ray.air import session
import os
import torch.optim as optim
import torch.nn as nn

from .models import FCN, FCNTemporalAttention, LSTM, LSTMSpatialAttention, LSTMTemporalAttention, LSTMSpatialTemporalAttention
from .data import Data


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


def train_setup(model, learning_rate, weight_decay):
    use_GPU = torch.cuda.is_available()
    if use_GPU:
        mode = {"name": "cuda", "device": torch.device("cuda")}
    else:
        mode = {"name": "cpu", "device": torch.device("cpu")}

    model.to(mode["device"])
    loss_function = nn.MSELoss().to(mode["device"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, loss_function, optimizer, mode


def train_model(config, checkpoint_dir=None): #TODO: DeprecationWarning: `checkpoint_dir` in `func(config, checkpoint_dir)` is being deprecated. To save and load checkpoint in trainable functions, please use the `ray.air.session` API

    #TODO: Change to save and load file if file exists in data folder. The app uploaded and does not download the file to data folder. Is is needed to download?
    # Set data file
    data_file = config['data_file']
    datetime_variable = config['datetime']

    # Load/create data
    data = Data(data_file, datetime_variable)
    data_loaders = data.prepare_data(**config['data'])

    # Map model names to classes
    model_classes = {
        "FCN": FCN,
        "FCNTemporalAttention": FCNTemporalAttention,
        "LSTMTemporalAttention": LSTMTemporalAttention,
        "LSTM": LSTM,
        "LSTMSpatialAttention": LSTMSpatialAttention,
        "LSTMSpatialTemporalAttention": LSTMSpatialTemporalAttention
    }

    # Get the model class based on the configuration
    model_name = config["model"]
    if model_name not in model_classes:
        raise ValueError(f"Invalid model name {model_name}. Possible model names are {list(model_classes.keys())}.")
    model_class = model_classes[model_name]

    # Set the input_size based on variables
    variables = config['data']['variables']
    config['model_arch']['input_size'] = len(variables) + 1 if variables else 1

    # Prepare training
    net = model_class(**config['model_arch'])
    num_epochs = config['num_epochs']
    net, loss_function, optimizer, mode = train_setup(net, **config['training'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    fit(net, loss_function, optimizer, data_loaders, num_epochs, mode, checkpoint_dir, use_amp=True)