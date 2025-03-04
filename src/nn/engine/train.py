from typing import Dict, List, Tuple, Union
from src.nn.engine.test import test
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (images, labels) in enumerate(dataloader):

        # Send data to target device
        images, labels = images.to(device), labels.to(device)

        # 1. Optimizer zero grad
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        # 2. Forward pass
        output = model(images)

        # 3. Calculate  and accumulate loss
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item()/len(output)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: Union[torch.nn.Module, Tuple],
          epochs: int, device: torch.device) -> Dict[str, List]:
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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
        f"\tTrain Epoch: {epoch + 1} \t"
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

    # Return the filled results at the end of the epochs
    return results