import os
import pandas as pd
from pathlib import Path

def split_dataset_into_clients_non_iid(num_clients=8, label_column="label"):
    """
    Generates a reproducible Non-IID split by forcing a deterministic sort 
    before dividing it sequentially among clients.
    """
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
    
    if label_column not in df.columns:
        print(f"CRITICAL: Label column '{label_column}' not found in metadata.")
        return
    
    test_df = df.sample(frac=0.2, random_state=42)
    test_df.to_csv(split_dir / "test.csv", index=False)
    print(f" - Saved {len(test_df)} samples to {split_dir / 'test.csv'} (Test set)")
    
    df = df.drop(test_df.index).reset_index(drop=True)
    
    print(f"Sorting data deterministically by '{label_column}' to induce non-IID skew...")
    
    df['_temp_id'] = df.index 
    secondary_sort_col = 'path' if 'path' in df.columns else '_temp_id'
    
    df = df.sort_values(by=[label_column, secondary_sort_col], kind="stable").reset_index(drop=True)
    
    if '_temp_id' in df.columns:
        df = df.drop(columns=['_temp_id'])
    
    total_samples = len(df)
    samples_per_client = total_samples // num_clients
    
    print(f"Total samples: {total_samples}")
    print(f"Samples per client: {samples_per_client}")

    # 4. Partition and save
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            client_df = df.iloc[start_idx:]
        else:
            client_df = df.iloc[start_idx : start_idx + samples_per_client]
            
        client_csv_path = split_dir / f"client_{i}.csv"
        client_df.to_csv(client_csv_path, index=False)
        
        unique_labels = client_df[label_column].unique()
        print(f" - Saved {len(client_df)} samples to {client_csv_path} (Contains labels: {unique_labels})")
    
    print(f"\nSuccess! Created {num_clients} Non-IID split files in {split_dir}")

if __name__ == "__main__":
    split_dataset_into_clients_non_iid(num_clients=8, label_column="finding")