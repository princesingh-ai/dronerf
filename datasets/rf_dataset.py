from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.preprocessing import normalize


class RFDataset(Dataset):
    """PyTorch dataset for RF IQ windows."""

    def __init__(self, dataset_path: str):
        self.samples = []
        dataset_path = Path(dataset_path)

        # Drone samples
        for file in (dataset_path / "drone").glob("*.npy"):
            windows = np.load(file, mmap_mode="r")
            for index in range(len(windows)):
                self.samples.append((file, index, 1))

        # Non-drone samples
        for file in (dataset_path / "non_drone").glob("*.npy"):
            windows = np.load(file, mmap_mode="r")
            for index in range(len(windows)):
                self.samples.append((file, index, 0))

    def __len__(self):
        """Return the total number of windows."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one IQ window and its label."""

        file, window_index, label = self.samples[index]
        windows = np.load(file, mmap_mode="r")
        window = windows[window_index]
        window = normalize(window)
        window = np.stack(
            (
                window.real,
                window.imag,
            ),
            axis=0,
        )
        window = torch.from_numpy(window.astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)
        return window, label