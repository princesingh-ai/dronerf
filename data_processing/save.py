from pathlib import Path
import numpy as np
from data_processing.loaders import load_signals
from data_processing.preprocessing import create_windows
from data_processing.split import split_dataset


def process_split(
    files: list[Path],
    output_dir: Path,
):
    """Load recordings, create windows, and save them."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        iq = load_signals(str(file))
        windows = create_windows(iq)
        output_path = output_dir / f"{file.stem}.npy"
        np.save(output_path, windows)


def create_dataset(
    drone_path: str,
    non_drone_path: str,
    output_dir: str,
):
    """Create train, validation, and test datasets."""

    drone_files = [
        file
        for file in Path(drone_path).glob("*.bin")
        if file.is_file()
    ]

    non_drone_files = [
        file
        for file in Path(non_drone_path).rglob("*.data")
        if file.is_file()
    ]

    (train_drone, val_drone, test_drone, train_non_drone, val_non_drone, test_non_drone,) = split_dataset(drone_files, non_drone_files,)
    output_dir = Path(output_dir)

    process_split(
        train_drone,
        output_dir / "train" / "drone",
    )

    process_split(
        train_non_drone,
        output_dir / "train" / "non_drone",
    )

    process_split(
        val_drone,
        output_dir / "validation" / "drone",
    )

    process_split(
        val_non_drone,
        output_dir / "validation" / "non_drone",
    )

    process_split(
        test_drone,
        output_dir / "test" / "drone",
    )

    process_split(
        test_non_drone,
        output_dir / "test" / "non_drone",
    )

if __name__ == "__main__":
    create_dataset(
        drone_path="drone_rf",
        non_drone_path="random_rf",
        output_dir="processed",
    )