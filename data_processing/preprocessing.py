import numpy as np

def normalize(iq: np.ndarray) -> np.ndarray:
    """
    Normalize IQ signals to the range [-1, 1].
    """
    peek = np.max(np.abs(iq))
    if peek == 0:
        return iq
    return iq / peek


def create_windows(iq: np.ndarray, window_size: int=4096, stride: int | None=None) -> np.ndarray:
    """
    Split a complex IQ signal into overlapping windows.
    """
    if stride is None:
        stride = window_size // 2  # Default: 50% overlap

    windows = []
    for start in range(0, len(iq) - window_size + 1, stride):
        windows.append(iq[start:start + window_size])
    return np.array(windows)