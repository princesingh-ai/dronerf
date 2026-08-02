from pathlib import Path

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