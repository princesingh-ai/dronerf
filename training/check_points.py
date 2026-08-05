from pathlib import Path
from training.config import DEVICE

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """Save a training checkpoint."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    path: str,
):
    """Load a training checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (
        model,
        optimizer,
        checkpoint["epoch"],
        checkpoint["loss"],
    )