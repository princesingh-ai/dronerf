import argparse
import json
import numpy as np
import torch
from collections import Counter
from pathlib import Path

from data_processing.loaders import load_signal
from data_processing.preprocessing import (
    create_windows,
    normalize,
)
from models.cnn1d import DroneCNN
from training.check_points import load_checkpoint
from utils.config import DEVICE, config
from data_processing.resample import resample_signal
from tqdm import tqdm


def predict(file_path: str, sample_rate: int | None = None, target_rate: int | None = None, show_progress: bool = True):
    """Predict the drone model of an RF recording."""

    iq = load_signal(file_path)

    if (sample_rate is not None and target_rate is not None and sample_rate != target_rate):
        print(f"Resampling: {sample_rate / 1e6:.0f} MSps -> " f"{target_rate / 1e6:.0f} MSps")
        print(f"Original samples : {len(iq):,}")

        iq = resample_signal(iq, original_rate=sample_rate, target_rate=target_rate)
        print(f"Resampled samples: {len(iq):,}")

    windows = create_windows(iq)
    
    idx_to_class = {0: "Noise/Background", 1: "Drone Detected"}

    model = DroneCNN().to(DEVICE)
    model, _, _, _ = load_checkpoint(model, None, "/Users/prince/projects/dronerf/checkpoints/best_model.pt")
    model.eval()

    all_preds = []

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
            
            pred = (torch.sigmoid(output) > 0.5).int().item()
            all_preds.append(pred)

    # Most common class predicted across all windows
    counter = Counter(all_preds)
    most_common_idx = counter.most_common(1)[0][0]
    prediction_str = idx_to_class[most_common_idx]
    
    agreement_ratio = counter[most_common_idx] / len(all_preds)

    print(f"Windows              : {len(all_preds)}")
    print(f"Agreement Ratio      : {agreement_ratio:.2%}")
    print(f"Prediction           : {prediction_str}")

    return {
        "windows": len(all_preds),
        "predicted_index": int(most_common_idx),
        "prediction_str": prediction_str,
        "agreement_ratio": agreement_ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF Drone Detection")

    parser.add_argument(
        "file",
        help="Path to RF recording",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=config["dataset"]["original_rate"],
        help="Sampling rate of the input recording (Hz)",
    )

    parser.add_argument(
        "--target-rate",
        type=int,
        default=config["dataset"]["target_rate"],
        help="Target sampling rate before inference (Hz)",
    )

    args = parser.parse_args()
    predict(args.file, sample_rate=args.sample_rate, target_rate=args.target_rate)