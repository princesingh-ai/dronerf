import torch.nn as nn

from training.config import DEVICE


def train_one_epoch(
    model,
    dataloader,
    criterion: nn.Module,
    optimizer,
):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for windows, labels in dataloader:

        windows = windows.to(DEVICE)

        labels = (
            labels
            .float()
            .unsqueeze(1)
            .to(DEVICE)
        )
        optimizer.zero_grad()
        outputs = model(windows)

        loss = criterion(
            outputs,
            labels,
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)