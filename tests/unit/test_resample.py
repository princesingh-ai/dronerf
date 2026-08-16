import numpy as np
import pytest
from data_processing.resample import resample_signal

def test_resample_identity():
    """Test that resampling to the same rate returns the original array exactly."""
    iq = np.random.randn(100) + 1j * np.random.randn(100)
    out = resample_signal(iq, original_rate=20_000_000, target_rate=20_000_000)
    np.testing.assert_array_equal(out, iq)

def test_resample_downsample_length():
    """Test that downsampling from 60Msps to 20Msps reduces length by a factor of 3."""
    iq = np.random.randn(300) + 1j * np.random.randn(300)
    out = resample_signal(iq, original_rate=60_000_000, target_rate=20_000_000)
    
    assert len(out) == 100
    assert np.iscomplexobj(out)

def test_resample_invalid_rates():
    """Test that negative or zero rates throw exceptions."""
    iq = np.random.randn(100) + 1j * np.random.randn(100)
    
    with pytest.raises(ValueError):
        resample_signal(iq, original_rate=0, target_rate=20_000_000)
        
    with pytest.raises(ValueError):
        resample_signal(iq, original_rate=60_000_000, target_rate=-10)

def test_resample_real_signal_fails():
    """Test that providing a non-complex signal throws an exception."""
    real_iq = np.random.randn(100)
    
    with pytest.raises(ValueError):
        resample_signal(real_iq, original_rate=60_000_000, target_rate=20_000_000)