import random

def split(files: list, train_ratio: float, val_ratio: float):
    """Split a list of recordings into train, validation, and test sets."""

    train_end = int(len(files) * train_ratio)
    val_end = train_end + int(len(files) * val_ratio)

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    return train, val, test

def split_dataset(drone_files: list, non_drone_files: list, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42,):
    """Split drone and non-drone recordings into train, validation, and test sets."""

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    random.seed(seed)

    drone_files = drone_files.copy()
    non_drone_files = non_drone_files.copy()

    random.shuffle(drone_files)
    random.shuffle(non_drone_files)

    train_drone, val_drone, test_drone = split(
        drone_files,
        train_ratio,
        val_ratio,
    )

    train_non_drone, val_non_drone, test_non_drone = split(
        non_drone_files,
        train_ratio,
        val_ratio,
    )

    return (
        train_drone,
        val_drone,
        test_drone,
        train_non_drone,
        val_non_drone,
        test_non_drone,
    )