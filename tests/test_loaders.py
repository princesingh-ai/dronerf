import numpy as np
import pytest

from data_processing.loaders import (
    load_bin,
    load_data,
    load_dat,
    load_signal,
)


BIN_FILES = [
    "drone_rf/DJI_inspire_2_2G.bin",
    "drone_rf/DJI_mavic_pro_2G.bin",
]

DATA_FILES = [
    "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T132400M058863.data",
    "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131300M012278.data",
    "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152100M004125.data",
    "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T150900M059227.data",
    "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191200M056171.data",
]

DAT_FILES = [
    "external_val/CLEAN/AIR_ON/AIR_0000_00.dat",
]


@pytest.mark.parametrize("file_path", BIN_FILES)
def test_load_bin(file_path):

    signal = load_bin(file_path)

    assert np.iscomplexobj(signal)
    assert signal.ndim == 1
    assert len(signal) > 0


@pytest.mark.parametrize("file_path", DATA_FILES)
def test_load_data(file_path):

    signal = load_data(file_path)

    assert np.iscomplexobj(signal)
    assert signal.ndim == 1
    assert len(signal) > 0


@pytest.mark.parametrize("file_path", DAT_FILES)
def test_load_dat(file_path):

    signal = load_dat(file_path)

    assert np.iscomplexobj(signal)
    assert signal.ndim == 1
    assert len(signal) > 0


@pytest.mark.parametrize(
    "file_path",
    BIN_FILES + DATA_FILES + DAT_FILES,
)
def test_load_signals(file_path):

    signal = load_signal(file_path)

    assert np.iscomplexobj(signal)
    assert signal.ndim == 1
    assert len(signal) > 0


def test_missing_file():

    with pytest.raises(FileNotFoundError):
        load_signal("does_not_exist.bin")


def test_invalid_extension(tmp_path):

    file = tmp_path / "signal.txt"
    file.write_text("hello")

    with pytest.raises(ValueError):
        load_signal(file)