from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(
    train_losses: list[float],
    validation_losses: list[float],
    output_path: str = "docs/images/loss_curve.png",
    ) -> None:
    """
    Save the training and validation loss curves.
    """

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        validation_losses,
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    output_path: str = "docs/images/confusion_matrix.png",
):
    """Save the confusion matrix as an image."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    matrix = np.array([
        [tp, fn],
        [fp, tn],
    ])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["Drone", "Non-Drone"],
    )

    plt.yticks(
        [0, 1],
        ["Drone", "Non-Drone"],
    )

    plt.colorbar()

    for row in range(2):
        for col in range(2):
            plt.text(col, row, matrix[row, col], ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()