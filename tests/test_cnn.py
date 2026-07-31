import torch
from models.cnn1d import DroneCNN


def test_model_output_shape():
    model = DroneCNN()
    x = torch.randn(8, 2, 4096)
    output = model(x)
    assert output.shape == (8, 1)


def test_model_output_dtype():
    model = DroneCNN()
    x = torch.randn(4, 2, 4096)
    output = model(x)
    assert output.dtype == torch.float32


def test_forward_pass():
    model = DroneCNN()
    x = torch.randn(2, 2, 4096)
    output = model(x)
    assert torch.isfinite(output).all()