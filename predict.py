import argparse

import numpy as np
import torch

from data_processing.loaders import load_bin
from data_processing.preprocessing import (
    create_windows,
    normalize,
)
from models.cnn1d import DroneCNN
from training.check_points import load_checkpoint
from training.config import (
    DEVICE,
    MODEL_PATH,
)


def predict(file_path: str):
    """Predict whether an RF recording contains a drone."""

    iq = load_bin(file_path)
    windows = create_windows(iq)
    model = DroneCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model, optimizer, _, _ = load_checkpoint(
        model,
        optimizer,
        MODEL_PATH,
    )
    model.eval()

    probabilities = []
    with torch.no_grad():

        for window in windows:
            window = normalize(window)
            window = np.stack(
                (
                    window.real,
                    window.imag,
                ),
                axis=0,
            )

            window = torch.from_numpy(
                window.astype(np.float32)
            )

            window = window.unsqueeze(0).to(DEVICE)
            output = model(window)
            probability = torch.sigmoid(output).item()
            probabilities.append(probability)

    average_probability = sum(probabilities) / len(probabilities)

    print(f"Windows              : {len(probabilities)}")
    print(f"Average Probability  : {average_probability:.4f}")

    if average_probability >= 0.5:
        print("Prediction           : Drone")
    else:
        print("Prediction           : Non-Drone")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RF Drone Detection",
    )

    parser.add_argument(
        "file",
        help="Path to RF recording",
    )

    args = parser.parse_args()

    predict(args.file)