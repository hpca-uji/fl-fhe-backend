from typing import Tuple, Union
import torch
import numpy as np
import tqdm

def test(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    
    # 1. Initialize lists to hold TENSORS for every batch
    all_logits = []
    all_labels = []
    activation = torch.nn.Sigmoid() 

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            # Fix class mismatch (ensure columns match)
            num_classes = min(output.shape[1], labels.shape[1])
            output = output[:, :num_classes]
            labels = labels[:, :num_classes]

            loss = loss_fn(output, labels)
            test_loss += loss.item()

            # 2. DETACH and store tensors (append to list)
            all_logits.append(output.detach().cpu())
            all_labels.append(labels.detach().cpu())

    # 3. CRITICAL: CONCATENATE ALL BATCHES AFTER THE LOOP
    # This turns your batches into one single tensor of 611 rows
    full_logits = torch.cat(all_logits, dim=0)
    full_labels = torch.cat(all_labels, dim=0)

    # 4. Convert to numpy ONCE after everything is aligned
    y_proba = activation(full_logits).numpy()
    y_true = full_labels.numpy()
    y_pred = (y_proba > 0.5).astype(float)

    # Metrics
    test_acc = (y_pred == y_true).mean() * 100
    test_loss /= len(dataloader)
    
    # This should now show (611, 25) (611, 25) in your terminal
    print(f"Shapes aligned: y_true={y_true.shape}, y_proba={y_proba.shape}")
    
    return test_loss, test_acc, y_pred, y_true, y_proba