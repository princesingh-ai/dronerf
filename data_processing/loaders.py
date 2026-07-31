import numpy as np

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

