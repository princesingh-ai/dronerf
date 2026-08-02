import torch
from training.plots import plot_confusion_matrix

from evaluation.confusion_matrix import (
    confusion_matrix,
    print_confusion_matrix,
)

from models.cnn1d import DroneCNN
from training.config import (
    DEVICE,
    MODEL_PATH,
    DATASET_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
)
from training.data_loader import create_dataloaders
from training.check_points import load_checkpoint
from training.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
)


def evaluate():
    """Evaluate the trained model on the test dataset."""

    _, _, test_loader = create_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = DroneCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model, optimizer, epoch, loss = load_checkpoint(
        model,
        optimizer,
        MODEL_PATH,
    )

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for windows, targets in test_loader:

            windows = windows.to(DEVICE)
            outputs = model(windows)
            predictions.append(outputs.cpu())
            labels.append(targets.float().unsqueeze(1))

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    print(f"Checkpoint Epoch : {epoch}")
    print(f"Validation Loss : {loss:.4f}")
    print(f"Accuracy        : {accuracy(predictions, labels):.4f}")
    print(f"Precision       : {precision(predictions, labels):.4f}")
    print(f"Recall          : {recall(predictions, labels):.4f}")
    print(f"F1 Score        : {f1_score(predictions, labels):.4f}")

    tp, tn, fp, fn = confusion_matrix(
    predictions,
    labels,
)

    print_confusion_matrix(
        tp,
        tn,
        fp,
        fn,
    )
    plot_confusion_matrix(
        tp,
        tn,
        fp,
        fn,
    )


if __name__ == "__main__":
    evaluate()