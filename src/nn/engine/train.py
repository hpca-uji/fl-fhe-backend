import os
import socket
from datetime import datetime
from typing import Dict, List, Tuple, Union

from rich import json
from src.core.util.load_config import LoadConfig
from src.nn.engine.test import test
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0

    for batch, data in enumerate(dataloader):
        if isinstance(data, dict):
            images, labels = data["img"], data["lab"]
        else:
            images, labels = data

        images, labels = images.to(device), labels.to(device)

        # Optimizer zero grad
        optimizer.zero_grad() 

        # Forward pass
        output = model(images)
        
        if output.shape[1] != labels.shape[1]:
            output = output[:, :labels.shape[1]]

        # Calculate and accumulate loss
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        
        # Calculate and accumulate accuracy
        if labels.ndim > 1 and labels.shape[1] > 1:
            # Multi-label accuracy (simple threshold at 0.5)
            preds = (torch.sigmoid(output) > 0.5).float()
            train_acc += (preds == labels).float().mean().item()
        else:
            # Standard Multi-class accuracy
            y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
            train_acc += (y_pred_class == labels).sum().item() / len(output)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: Union[torch.nn.Module, Tuple],
          epochs: int, device: torch.device, server_round: int = 1, client_id: str = "0") -> Dict[str, List]:
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    config = LoadConfig("pyproject.toml")
    stats = config.get_stats_config()
    nn = config.get_nn_config()
    he = config.get_he_config()
    
    device_name = socket.gethostname()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a directory for this client's results if it doesn't exist
    client_results_dir = f"results/client_{client_id}"
    os.makedirs(client_results_dir, exist_ok=True)
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), colour="BLUE"):
        # Select functions based on the task
        train_step_fn = train_step
        test_fn = test

        # Perform training and validation
        train_loss, train_acc = train_step_fn(model=model, dataloader=train_dataloader, loss_fn=loss_fn,
                                            optimizer=optimizer, device=device)
        val_loss, val_acc, *_ = test_fn(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)        
        
        # Print out what's happening
        print(
        f"\tRound: {server_round} \t"
        f"Train Epoch: {epoch + 1} \t"
        f"Train_loss: {train_loss:.4f} | "
        f"Train_acc: {train_acc:.4f} % | "
        f"Validation_loss: {val_loss:.4f} | "
        f"Validation_acc: {val_acc:.4f} %"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # Save intermediate results to a single file in a csv format, updating it each epoch
        results_file = os.path.join(client_results_dir, f"results_{device_name}.csv")
        file_exists = os.path.isfile(results_file)
        with open(results_file, "a") as f:
            if not file_exists:
                f.write("date,model,fhe_lib,round,epoch,device_name,train_loss,train_acc,val_loss,val_acc\n")
            f.write(f"{current_date},{nn['model']},{he['library']},{server_round},{epoch + 1},{device_name},{train_loss},{train_acc},{val_loss},{val_acc}\n")
                

    # Return the filled results at the end of the epochs
    return results