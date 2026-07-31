import torch
import torch.nn as nn
from models.cnn1d import DroneCNN


def test_forward_pass():
    """Model should perform a forward pass."""
    model = DroneCNN()
    windows = torch.randn(8, 2, 4096)
    outputs = model(windows)

    assert outputs.shape == (8, 1)


def test_loss_computation():
    """Loss should be computed successfully."""
    model = DroneCNN()
    criterion = nn.BCEWithLogitsLoss()
    windows = torch.randn(8, 2, 4096)

    labels = torch.randint(
        0,
        2,
        (8, 1),
    ).float()
    outputs = model(windows)

    loss = criterion(
        outputs,
        labels,
    )
    assert torch.isfinite(loss)


def test_backward_pass():
    """Backward pass should compute gradients."""
    model = DroneCNN()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    windows = torch.randn(8, 2, 4096)

    labels = torch.randint(
        0,
        2,
        (8, 1),
    ).float()
    outputs = model(windows)

    loss = criterion(
        outputs,
        labels,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert loss.item() > 0


def test_train_mode():
    """Model should be in training mode."""
    model = DroneCNN()
    model.train()
    assert model.training