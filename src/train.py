import torch
from ray import tune
from tqdm import tqdm
from ray.air import session
import os

def fit(net, loss_function, optimizer, data_loader, num_epochs, mode, checkpoint_dir, use_amp=False):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # Mixed-precision support for compatible GPUs
    losses = {"train": [], "val": [], "test": []}  # Add this line to store losses
    for epoch in range(num_epochs):
        if epoch < num_epochs - 1:
            keys = ["train", "val"]
        else:
            keys = ["train", "val", "test"]
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

            # Report results to Ray Tune
            if key == "train":
                tune.report(train_loss=dataset_loss)
                losses[key].append(dataset_loss)
            elif key == "val":
                # Update learning rate
                tune.report(val_loss=dataset_loss)
                losses[key].append(dataset_loss)
            else:
                tune.report(test_loss=dataset_loss)
                losses[key].append(dataset_loss)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

    return net