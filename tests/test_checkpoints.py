from pathlib import Path
import torch
from models.cnn1d import DroneCNN
from training.check_points import (
    load_checkpoint,
    save_checkpoint,
)


CHECKPOINT_PATH = "checkpoints/test_checkpoint.pt"


def test_save_checkpoint():
    """Checkpoint should be saved successfully."""

    model = DroneCNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        loss=0.1234,
        path=CHECKPOINT_PATH,
    )

    assert Path(CHECKPOINT_PATH).exists()


def test_load_checkpoint():
    """Checkpoint should be loaded successfully."""

    model = DroneCNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        loss=0.1234,
        path=CHECKPOINT_PATH,
    )

    model, optimizer, epoch, loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=CHECKPOINT_PATH,
    )

    assert epoch == 5
    assert loss == 0.1234