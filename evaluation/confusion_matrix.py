import torch


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor):
    """Compute the confusion matrix."""

    predictions = (torch.sigmoid(predictions) >= 0.5).int()
    labels = labels.int()

    tp = ((predictions == 1) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    return tp, tn, fp, fn


def print_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
):
    """Print the confusion matrix."""

    print()
    print("Confusion Matrix")
    print("------------------------------")
    print(f"{'':15}Predicted")
    print(f"{'':15}Drone   Non-Drone")
    print(f"{'Drone':15}{tp:<8}{fn}")
    print(f"{'Non-Drone':15}{fp:<8}{tn}")