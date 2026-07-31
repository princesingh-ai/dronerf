import torch

from training.data_loader import create_dataloaders


def test_dataloader():

    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset_path="processed",
        batch_size=32,
    )
    windows, labels = next(iter(train_loader))

    assert isinstance(windows, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert windows.ndim == 3
    assert labels.ndim == 1
    assert windows.shape[1:] == (2, 4096)
    assert labels.dtype == torch.long

    assert len(train_loader) > 0
    assert len(validation_loader) > 0
    assert len(test_loader) > 0