import torch
import torch.nn as nn
from training.config import DEVICE


def validate(
    model,
    dataloader,
    criterion: nn.Module,
):
    """Evaluate the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():

        for windows, labels in dataloader:
            windows = windows.to(DEVICE)
            labels = (
                labels
                .float()
                .unsqueeze(1)
                .to(DEVICE)
            )
            outputs = model(windows)

            loss = criterion(
                outputs,
                labels,
            )
            running_loss += loss.item()

    validation_loss = running_loss / len(dataloader)

    return validation_loss