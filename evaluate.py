import torch
import json
from pathlib import Path
from training.plots import plot_confusion_matrix

from evaluation.confusion_matrix import (
    calculate_confusion_matrix,
    print_confusion_matrix,
)

from models.cnn1d_multiclass import MultiClassDroneCNN
from utils.config import DEVICE, config
from training.data_loader import create_dataloaders
from training.check_points import load_checkpoint
from training.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
)

# This evaluation script is now fully agnostic to the number of drone models.
# It uses the mapping file to reconstruct the exact labels and prints macro/weighted metrics.
def evaluate():
    """Evaluate the trained multi-class model on the test dataset."""

    mapping_path = Path(config["dataset"]["mapping_path"])
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing {mapping_path}. Run save.py first!")
        
    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
        
    # Create an ordered list of class names based on their integer ID
    class_names = ["" for _ in range(len(class_mapping))]
    for name, class_id in class_mapping.items():
        class_names[class_id] = name

    num_classes = len(class_names)

    _, _, test_loader = create_dataloaders(
        dataset_path=config["dataset"]["processed_path"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    model = MultiClassDroneCNN(num_classes=num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model, optimizer, epoch, loss = load_checkpoint(
        model,
        optimizer,
        config["training"]["model_path"],
    )

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for windows, targets in test_loader:
            windows = windows.to(DEVICE)
            outputs = model(windows)
            
            # For multi-class, we extract the predicted class index by taking the argmax over the logits
            preds = torch.argmax(outputs, dim=1)
            
            predictions.append(preds.cpu())
            labels.append(targets.cpu())

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    print(f"Checkpoint Epoch : {epoch}")
    print(f"Validation Loss : {loss:.4f}")
    
    # We output macro averages to treat every drone model equally, preventing the 
    # background class (which is huge) from dominating the accuracy metric.
    print(f"Accuracy        : {accuracy(predictions, labels):.4f}")
    print(f"Precision (Mac) : {precision(predictions, labels, average='macro'):.4f}")
    print(f"Recall (Mac)    : {recall(predictions, labels, average='macro'):.4f}")
    print(f"F1 Score (Mac)  : {f1_score(predictions, labels, average='macro'):.4f}")
    print(f"F1 Score (Wgt)  : {f1_score(predictions, labels, average='weighted'):.4f}")

    # Generate and plot the NxN multi-class matrix
    matrix = calculate_confusion_matrix(predictions, labels)

    print_confusion_matrix(matrix, class_names)
    plot_confusion_matrix(matrix, class_names)


if __name__ == "__main__":
    evaluate()