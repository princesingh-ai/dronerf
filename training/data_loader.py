from torch.utils.data import DataLoader
from datasets.rf_dataset import RFDataset


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """Create train, validation, and test dataloaders."""

    train_dataset = RFDataset(f"{dataset_path}/train")
    validation_dataset = RFDataset(f"{dataset_path}/validation")
    test_dataset = RFDataset(f"{dataset_path}/test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )