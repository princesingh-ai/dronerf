import torch
import torch.nn as nn

from models.cnn1d import DroneCNN
from training.config import DEVICE
from training.data_loader import create_dataloaders
from training.validate import validate


def test_validation_returns_float():
    """Validation should return a floating-point loss."""

    _, validation_loader, _ = create_dataloaders(
        dataset_path="processed",
        batch_size=8,
    )

    model = DroneCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    validation_loss = validate(
        model,
        validation_loader,
        criterion,
    )

    assert isinstance(validation_loss, float)


def test_validation_loss_positive():
    """Validation loss should be non-negative."""

    _, validation_loader, _ = create_dataloaders(
        dataset_path="processed",
        batch_size=8,
    )

    model = DroneCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    validation_loss = validate(
        model,
        validation_loader,
        criterion,
    )

    assert validation_loss >= 0.0


def test_model_in_eval_mode():
    """Model should be in evaluation mode after validation."""

    _, validation_loader, _ = create_dataloaders(
        dataset_path="processed",
        batch_size=8,
    )

    model = DroneCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    validate(
        model,
        validation_loader,
        criterion,
    )

    assert not model.training