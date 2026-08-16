import torch
import numpy as np
from evaluation.confusion_matrix import calculate_confusion_matrix

def test_confusion_matrix():
    """Test that the confusion matrix outputs an NxN matrix with correct order."""
    
    # Let's say we have 3 classes (0, 1, 2)
    labels = torch.tensor([0, 1, 2, 2, 0])
    predictions = torch.tensor([0, 1, 1, 2, 2])
    
    # 0 -> [1, 0, 1] (predicted 0 once, predicted 2 once)
    # 1 -> [0, 1, 0] (predicted 1 once)
    # 2 -> [0, 1, 1] (predicted 1 once, predicted 2 once)
    
    matrix = calculate_confusion_matrix(predictions, labels)
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    
    # Check row 0
    assert (matrix[0] == [1, 0, 1]).all()
    # Check row 1
    assert (matrix[1] == [0, 1, 0]).all()
    # Check row 2
    assert (matrix[2] == [0, 1, 1]).all()
