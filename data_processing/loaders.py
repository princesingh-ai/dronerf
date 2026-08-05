import numpy as np
from pathlib import Path

def load_bin(path) -> np.ndarray:
    """
    Load binary data from a file and return IQ signals.
    """
    raw_data = np.fromfile(path, dtype=np.int16)

    if len(raw_data) % 2 != 0:
        raise ValueError(
            f"{path} does not contain an even number of float32 values."
        )

    # convert to complex numbers
    I = raw_data[0::2].astype(np.float32)
    Q = raw_data[1::2].astype(np.float32)

    iq = I + 1j * Q

    return iq

def load_data(path) -> np.ndarray:
    """
    Load data from a .data file and return IQ signals.
    """
    raw_data = np.fromfile(path, dtype=np.int16)

    # convert to complex numbers
    I = raw_data[0::2].astype(np.float32)
    Q = raw_data[1::2].astype(np.float32)

    iq = I + 1j * Q

    return iq

def load_dat(path) -> np.ndarray:
    """
    Load data from a .dat file and return IQ signals.
    """
    raw_data = np.fromfile(path, dtype=np.float32)

    if len(raw_data) % 2 != 0:
        raise ValueError(
            f"{path} does not contain an even number of float32 values."
        )

    # convert to complex numbers
    I = raw_data[0::2].astype(np.float32)
    Q = raw_data[1::2].astype(np.float32)

    iq = I + 1j * Q

    return iq

def load_signal(path) -> np.ndarray:
    """"
    Load signals from a file and return IQ signals.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension == ".bin":
        return load_bin(path)
    elif extension == ".data":
        return load_data(path)
    elif extension == ".dat":
        return load_dat(path)

    raise ValueError(f"Unsupported file extension: {extension}")
