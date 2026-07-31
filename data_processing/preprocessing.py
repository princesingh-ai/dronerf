import numpy as np

def normalize(iq: np.ndarray) -> np.ndarray:
    """
    Normalize IQ signals to the range [-1, 1].
    """
    peek = np.max(np.abs(iq))
    if peek == 0:
        return iq
    return iq / peek