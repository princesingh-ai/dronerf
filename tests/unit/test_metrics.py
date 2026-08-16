import torch
import numpy as np
from training.metrics import accuracy, precision, recall, f1_score

def test_metrics():
    """Test the scikit-learn based multi-class metrics using known deterministic synthetic predictions."""
    
    # 3 classes: 0, 1, 2
    # Let's say:
    # labels:      [0, 0, 1, 1, 2, 2]
    # predictions: [0, 1, 1, 1, 2, 0]
    
    # Class 0: True (1), False Positives (1), False Negatives (1) -> Precision 1/2, Recall 1/2
    # Class 1: True (2), False Positives (1), False Negatives (0) -> Precision 2/3, Recall 2/2
    # Class 2: True (1), False Positives (0), False Negatives (1) -> Precision 1/1, Recall 1/2
    
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    predictions = torch.tensor([0, 1, 1, 1, 2, 0], dtype=torch.long)
    
    # Accuracy: 4/6 = 0.666...
    acc = accuracy(predictions, labels)
    assert np.isclose(acc, 4/6)
    
    # Precision Macro: (0.5 + 0.666... + 1.0) / 3 = 0.7222...
    p_mac = precision(predictions, labels, average="macro")
    assert np.isclose(p_mac, (0.5 + 2/3 + 1.0) / 3)
    
    # Recall Macro: (0.5 + 1.0 + 0.5) / 3 = 0.666...
    r_mac = recall(predictions, labels, average="macro")
    assert np.isclose(r_mac, (0.5 + 1.0 + 0.5) / 3)
    
    # F1 Score Macro calculation validation
    f1_mac = f1_score(predictions, labels, average="macro")
    assert f1_mac > 0.0 and f1_mac < 1.0
    
    # F1 Score Weighted calculation validation
    f1_wgt = f1_score(predictions, labels, average="weighted")
    assert f1_wgt > 0.0 and f1_wgt < 1.0