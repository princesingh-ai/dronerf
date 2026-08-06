import numpy as np
import pytest

from data_processing.resample import resample_signal

def test_same_sampling_rate():
    iq = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)

    output = resample_signal(iq, original_rate=120_000_000, target_rate=120_000_000)
    assert np.array_equal(iq, output)


def test_60_to_120():
    iq = (np.random.randn(6000) + 1j * np.random.randn(6000)).astype(np.complex64)

    output = resample_signal(iq, original_rate=60_000_000, target_rate=120_000_000)
    assert abs(len(output) - 12000) <= 2


def test_120_to_60():
    iq = (np.random.randn(12000) + 1j * np.random.randn(12000)).astype(np.complex64)

    output = resample_signal(iq, original_rate=120_000_000, target_rate=60_000_000)
    assert abs(len(output) - 6000) <= 2


def test_200_to_120():
    iq = (np.random.randn(20000) + 1j * np.random.randn(20000)).astype(np.complex64)

    output = resample_signal(iq, original_rate=200_000_000, target_rate=120_000_000)

    expected = int(len(iq) * 120 / 200)
    assert abs(len(output) - expected) <= 2


def test_complex_output():
    iq = (np.random.randn(5000) + 1j * np.random.randn(5000)).astype(np.complex64)

    output = resample_signal(iq, original_rate=60_000_000, target_rate=120_000_000)
    assert np.iscomplexobj(output)


def test_invalid_original_rate():
    iq = (np.random.randn(1000) + 1j * np.random.randn(1000))

    with pytest.raises(ValueError):
        resample_signal(iq, original_rate=0, target_rate=120_000_000)


def test_invalid_target_rate():
    iq = (np.random.randn(1000) + 1j * np.random.randn(1000))

    with pytest.raises(ValueError):
        resample_signal(iq ,original_rate=120_000_000, target_rate=0)


def test_non_complex_input():
    signal = np.random.randn(1000)

    with pytest.raises(ValueError):
        resample_signal(signal, original_rate=60_000_000, target_rate=120_000_000)