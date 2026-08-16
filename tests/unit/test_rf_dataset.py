import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from datasets.rf_dataset import RFDataset

def test_rf_dataset_synthetic():
    """Test the RFDataset using tiny synthetic .npy files to ensure correct label mapping and dtype."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # 1. Create a fake class_mapping.json in the parent directory of our dataset split
        mapping = {"non_drone": 0, "drone_alpha": 1}
        with open(base_dir / "class_mapping.json", "w") as f:
            json.dump(mapping, f)
            
        # 2. Create the dataset split directory (e.g. "train")
        split_dir = base_dir / "train"
        split_dir.mkdir()
        
        # 3. Create synthetic classes and data
        for class_name, label_id in mapping.items():
            class_dir = split_dir / class_name
            class_dir.mkdir()
            
            # Create a mock .npy file with 3 windows of size 4096 (complex)
            mock_windows = np.random.randn(3, 4096) + 1j * np.random.randn(3, 4096)
            np.save(class_dir / f"mock_{class_name}.npy", mock_windows)
            
        # 4. Instantiate the Dataset
        dataset = RFDataset(str(split_dir))
        
        # We should have 3 windows * 2 classes = 6 samples
        assert len(dataset) == 6
        
        # 5. Test __getitem__
        window_tensor, label_tensor = dataset[0]
        
        # Shape should be [2, 4096]
        assert window_tensor.shape == (2, 4096)
        assert window_tensor.dtype == torch.float32
        
        # Label should be torch.long (crucial for CrossEntropyLoss)
        assert label_tensor.dtype == torch.long
        
        # Ensure the labels are within our expected mapping
        all_labels = [dataset[i][1].item() for i in range(len(dataset))]
        assert set(all_labels) == {0, 1}
        assert all_labels.count(0) == 3
        assert all_labels.count(1) == 3
