import torch

from training.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
)


def test_perfect_prediction():
    predictions = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    assert accuracy(predictions, labels) == 1.0
    assert precision(predictions, labels) == 1.0
    assert recall(predictions, labels) == 1.0
    assert f1_score(predictions, labels) == 1.0


def test_completely_wrong_prediction():
    predictions = torch.tensor([[-5.0], [5.0]])
    labels = torch.tensor([[1.0], [0.0]])
    assert accuracy(predictions, labels) == 0.0