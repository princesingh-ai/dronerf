from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(iq: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample a complex IQ signal to a new sampling rate.

    Parameters
    ----------
    iq : np.ndarray
        Complex IQ signal.

    original_rate : int
        Original sampling rate in samples per second.

    target_rate : int
        Desired sampling rate in samples per second.

    Returns
    -------
    np.ndarray
        Resampled complex IQ signal.
    """

    if original_rate <= 0:
        raise ValueError("original_rate must be positive.")

    if target_rate <= 0:
        raise ValueError("target_rate must be positive.")

    if not np.iscomplexobj(iq):
        raise ValueError("Input signal must be complex.")

    if original_rate == target_rate:
        return iq

    # Find the greatest common divisor (GCD) of the original and target sample rates.
    # This reduces the upsampling and downsampling values to the smallest possible numbers.
    # Then use these values to resample the IQ data from the original rate to the target rate.
    factor = gcd(original_rate, target_rate)

    up = target_rate // factor
    down = original_rate // factor

    return resample_poly(iq, up=up, down=down)