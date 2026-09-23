
from src.core.util.split_dataset import split_dataset_non_iid_variable_samples
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np
import joblib
import os
import shutil

num_clients = 100

transform = transforms.ToTensor()
cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

x_train = np.stack([np.array(img) for img, _ in cifar100])  # shape (50000, 32, 32, 3)
y_train = np.array([label for _, label in cifar100]).flatten()  # shape (50000,) 

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

clients = split_dataset_non_iid_variable_samples(
    x_train,
    y_train,
    num_clients=num_clients,
    alpha=0.1,
    num_classes=100,
    min_size=0,
    max_size=x_train.shape[0] # num_clients,
)

# Remove directories if they exist
for i in range(num_clients):
    dir_path = os.path.join("data", str(i))
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)

# Create a csv with the number of samples per client
with open('num_samples_per_client.txt', 'w') as f:
    for i, (x, y) in enumerate(clients):
        f.write(f"Client {i}: {len(x)} samples\n")

for i, (x, y) in enumerate(clients):
    # Save each client's dataset to a separate file
    filename = f"client_{i}.pkl"
    path = os.path.join("data", str(i))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    with open(os.path.join(path, filename), 'wb') as f:
        joblib.dump((x, y), f)