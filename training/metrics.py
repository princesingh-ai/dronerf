import torch


def accuracy(predictions, labels):
    """Compute classification accuracy."""
    predictions = (torch.sigmoid(predictions) >= 0.5).float()
    correct = (predictions == labels).sum().item()
    
    return correct / len(labels)


def precision(predictions, labels):
    """Compute precision."""
    predictions = (torch.sigmoid(predictions) >= 0.5).float()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(predictions, labels):
    """Compute recall."""
    predictions = (torch.sigmoid(predictions) >= 0.5).float()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(predictions, labels):
    """Compute F1-score."""
    p = precision(predictions, labels)
    r = recall(predictions, labels)

    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)