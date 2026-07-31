import numpy as np

def load_bin(path) -> np.ndarray:
    """
    Load binary data from a file and return the real and imaginary parts of the signal.
    """
    raw_data = np.fromfile(path, dtype=np.int16)

    # convert to complex numbers
    I = raw_data[0::2].astype(np.float32)
    Q = raw_data[1::2].astype(np.float32)

    iq = I + 1j * Q

    return iq