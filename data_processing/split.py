import random
from typing import Dict, List, Tuple
from pathlib import Path

# Helper function to split a single list of files into train, val, and test.
def split(files: list, train_ratio: float, val_ratio: float):
    # Ensure at least 1 file goes to training if the class has any files at all
    train_end = max(1, int(len(files) * train_ratio)) if len(files) > 0 else 0
    val_end = train_end + int(len(files) * val_ratio)

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    return train, val, test

# We are switching from a hardcoded drone/non-drone split to a generic multi-class split.
# This ensures that every individual drone model (and non-drone) gets the exact 80/10/10 split,
# avoiding situations where a specific drone ends up entirely in the test set.
def split_dataset(
    class_files: Dict[str, List[Path]], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    seed: int = 42
) -> Dict[str, Dict[str, List[Path]]]:
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    random.seed(seed)

    splits = {
        "train": {},
        "validation": {},
        "test": {}
    }

    # Iterate through each class independently to maintain stratification
    for class_name, files in class_files.items():
        files_copy = files.copy()
        random.shuffle(files_copy)

        train_files, val_files, test_files = split(files_copy, train_ratio, val_ratio)

        splits["train"][class_name] = train_files
        splits["validation"][class_name] = val_files
        splits["test"][class_name] = test_files

    return splits