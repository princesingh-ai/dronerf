import argparse

import numpy as np
import torch

from data_processing.loaders import load_signal
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

from tqdm import tqdm

from data_processing.resample import resample_signal

def predict(file_path: str, sample_rate: int | None = None, target_rate: int | None = None, show_progress: bool = True):
    """Predict whether an RF recording contains a drone."""

    iq = load_signal(file_path)

    if (sample_rate is not None and target_rate is not None and sample_rate != target_rate):

        print(f"Resampling: {sample_rate / 1e6:.0f} MSps -> " f"{target_rate / 1e6:.0f} MSps")
        print(f"Original samples : {len(iq):,}")

        iq = resample_signal(iq, original_rate=sample_rate, target_rate=target_rate)
        print(f"Resampled samples: {len(iq):,}")

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

    iterator = tqdm(windows, desc="Predicting", unit="window") if show_progress else windows

    with torch.no_grad():

        for window in iterator:
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

    prediction = ("Drone" if average_probability >= 0.5 else "Non-Drone")

    print(f"Windows              : {len(probabilities)}")
    print(f"Average Probability  : {average_probability:.4f}")
    print(f"Prediction           : {prediction}")

    return {
        "windows": len(probabilities),
        "average_probability": float(average_probability),
        "prediction": prediction,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RF Drone Detection",
    )

    parser.add_argument(
        "file",
        help="Path to RF recording",
    )

    parser.add_argument(
    "--sample-rate",
    type=int,
    default=None,
    help="Sampling rate of the input recording (Hz)")

    parser.add_argument(
    "--target-rate",
    type=int,
    default=None,
    help="Target sampling rate before inference (Hz)")

    args = parser.parse_args()

    predict(args.file, sample_rate=args.sample_rate, target_rate=args.target_rate)