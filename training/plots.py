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
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str = "docs/images/confusion_matrix.png",
):
    """Save the multi-class NxN confusion matrix as an image."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # We dynamically size the figure based on the number of classes so labels don't get squished
    fig_size = max(6, len(class_names) * 1.5)
    plt.figure(figsize=(fig_size, fig_size - 1))
    
    plt.imshow(matrix, cmap="Blues")

    plt.title("Multi-Class Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    # Rotate the x-axis labels so long drone names don't overlap
    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(len(class_names)),
        class_names,
    )

    plt.colorbar()

    # Add the text annotations inside the boxes
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            plt.text(col, row, matrix[row, col], ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()