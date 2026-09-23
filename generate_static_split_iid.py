import os
import pandas as pd
from pathlib import Path

def split_dataset_into_clients_iid(num_clients=10, seed=42):
    """
    Generates an IID split by shuffling the dataset and dividing it equally.
    """
    # 1. Create directory structure
    data_dir = Path("data/")
    split_dir = Path("data/splits")
    data_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
 
    csv_path = data_dir / "metadata.csv"
    if not csv_path.exists():
        print(f"CRITICAL: metadata.csv not found in {data_dir}")
        return

    print("Loading metadata...")
    df = pd.read_csv(csv_path)
    
    # 2. Shuffle data to ensure IID (Independent and Identically Distributed)
    # Using a fixed seed ensures that every time you run this, clients get the same data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 3. Calculate split sizes
    total_samples = len(df)
    samples_per_client = total_samples // num_clients
    
    print(f"Total samples: {total_samples}")
    print(f"Samples per client: {samples_per_client}")

    # 4. Partition and save
    for i in range(num_clients):
        # Handle the last client by giving them all remaining samples (to avoid rounding loss)
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            client_df = df.iloc[start_idx:]
        else:
            client_df = df.iloc[start_idx : start_idx + samples_per_client]
            
        client_csv_path = split_dir / f"client_{i}.csv"
        client_df.to_csv(client_csv_path, index=False)
        print(f" - Saved {len(client_df)} samples to {client_csv_path}")
    
    print(f"\nSuccess! Created {num_clients} IID split files in {split_dir}")

if __name__ == "__main__":
    split_dataset_into_clients_iid(num_clients=5)
