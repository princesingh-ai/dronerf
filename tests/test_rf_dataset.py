import torch
from datasets.rf_dataset import RFDataset


def test_dataset_not_empty():
    """Dataset should contain at least one sample."""
    dataset = RFDataset("processed/train")
    assert len(dataset) > 0


def test_dataset_returns_tensor():
    """Dataset should return tensors."""
    dataset = RFDataset("processed/train")
    window, label = dataset[0]
    assert isinstance(window, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_window_shape():
    """Each window should have shape (2, 4096)."""
    dataset = RFDataset("processed/train")
    window, _ = dataset[0]
    assert window.shape == (2, 4096)


def test_window_dtype():
    """Window tensor should be float32."""
    dataset = RFDataset("processed/train")
    window, _ = dataset[0]
    assert window.dtype == torch.float32


def test_label_dtype():
    """Label should be an integer tensor."""
    dataset = RFDataset("processed/train")
    _, label = dataset[0]
    assert label.dtype == torch.long


def test_label_value():
    """Label should be either 0 or 1."""
    dataset = RFDataset("processed/train")
    _, label = dataset[0]
    assert label.item() in [0, 1]


def test_normalization():
    """Window should be normalized."""
    dataset = RFDataset("processed/train")
    window, _ = dataset[0]
    assert torch.max(torch.abs(window)) <= 1.0 + 1e-6


def test_multiple_samples():
    """Multiple samples should be readable."""
    dataset = RFDataset("processed/train")
    indices = [0, len(dataset) // 2, len(dataset) - 1]

    for index in indices:
        window, label = dataset[index]
        assert window.shape == (2, 4096)
        assert label.item() in [0, 1]

def test_contains_both_classes():
    """Dataset should contain drone and non-drone samples."""
    dataset = RFDataset("processed/train")
    labels = set()

    for _, _, label in dataset.samples:
        labels.add(label)
    assert labels == {0, 1}