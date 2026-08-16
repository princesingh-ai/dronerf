from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sklearn_f1_score
import torch

# We are switching from manual binary metrics to scikit-learn's robust multi-class metrics.
# These functions now expect `predictions` and `labels` to be 1D tensors of integer class indices.

def accuracy(predictions, labels):
    """Compute classification accuracy."""
    # Move to CPU and numpy for scikit-learn
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    return accuracy_score(label_np, pred_np)


def precision(predictions, labels, average='macro'):
    """Compute precision (macro by default for multi-class)."""
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    # zero_division=0 prevents warnings if a class is never predicted
    return precision_score(label_np, pred_np, average=average, zero_division=0)


def recall(predictions, labels, average='macro'):
    """Compute recall (macro by default for multi-class)."""
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    return recall_score(label_np, pred_np, average=average, zero_division=0)


def f1_score(predictions, labels, average='macro'):
    """Compute F1-score (macro by default for multi-class)."""
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    return sklearn_f1_score(label_np, pred_np, average=average, zero_division=0)