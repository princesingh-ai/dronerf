import torch
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import numpy as np

# We're moving from a hardcoded 2x2 matrix (TP, TN, FP, FN) to an NxN matrix 
# that handles any number of drone classes dynamically using scikit-learn.

def calculate_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor):
    """Compute the multi-class confusion matrix."""
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    
    return sklearn_confusion_matrix(label_np, pred_np)


def print_confusion_matrix(matrix: np.ndarray, class_names: list[str]):
    """Print the NxN confusion matrix to the console."""
    
    print("\nConfusion Matrix")
    print("-" * 50)
    
    # We want a nice dynamic column header
    header = f"{'':20}" + "".join([f"{name[:8]:>10}" for name in class_names])
    print(header)
    
    # Print each row with its true label class name
    for i, row in enumerate(matrix):
        row_str = "".join([f"{val:>10}" for val in row])
        print(f"{class_names[i][:18]:<20}{row_str}")