import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.models as models
from pathlib import Path

class LazyNIHDataset(Dataset):
    def __init__(self, img_path, transform=None):
        """
        Memory-efficient dataset loader for NIH Chest X-rays.
        Only stores the filenames in memory.
        """
        self.img_path = Path(img_path)
        
        # Ensure the path exists or create it (though data should already be there)
        if not self.img_path.exists():
            self.img_path.mkdir(parents=True, exist_ok=True)
            print(f"Warning: Created empty directory at {img_path}. Place NIH images here.")

        # Get all image names - os.listdir is more memory efficient than loading a 100k row CSV
        self.img_names = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load single image only when needed
        img_name = self.img_names[idx]
        img_path = self.img_path / img_name
        
        # Use a context manager to ensure the file handle is closed immediately
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # NIH has 14 labels. In a truly lazy setup without the CSV, we use dummy labels 
        # unless you provide a small per-client mapping file.
        # For simulation, we return a zero-tensor of size 14.
        labels = torch.zeros(14) 
        
        return image, labels