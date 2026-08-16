import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.preprocessing import normalize

# We are switching to a dynamic multi-class dataset loader. 
# It will read the class_mapping.json to figure out which integer label to assign to each folder.
class RFDataset(Dataset):
    """PyTorch dataset for multi-class RF IQ windows."""

    def __init__(self, dataset_path: str):
        self.samples = []
        dataset_path = Path(dataset_path)
        
        # The class mapping is stored in the parent directory (e.g., 'processed/')
        mapping_path = dataset_path.parent / "class_mapping.json"
        with open(mapping_path, "r") as f:
            self.class_mapping = json.load(f)

        # Iterate through every class folder in this split (e.g., processed/train/DJI_inspire_2)
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in self.class_mapping:
                print(f"Warning: Found folder {class_name} but it's not in class_mapping.json")
                continue
                
            label = self.class_mapping[class_name]
            
            # Load all the .npy window files inside this class folder
            for file in class_dir.glob("*.npy"):
                windows = np.load(file, mmap_mode="r")
                for index in range(len(windows)):
                    self.samples.append((file, index, label))

    def __len__(self):
        """Return the total number of windows."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one IQ window and its integer label."""
        file, window_index, label_int = self.samples[index]
        
        # Lazily load the specific window from the memory-mapped numpy array
        windows = np.load(file, mmap_mode="r")
        window = windows[window_index]
        window = normalize(window)
        
        # Stack Real and Imaginary parts into 2 channels
        window = np.stack((window.real, window.imag), axis=0)
        window_tensor = torch.from_numpy(window.astype(np.float32))
        
        # We must return torch.long for CrossEntropyLoss
        label_tensor = torch.tensor(label_int, dtype=torch.long)
        
        return window_tensor, label_tensor