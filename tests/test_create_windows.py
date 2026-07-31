import numpy as np

from data_processing.loaders import load_bin, load_data
from data_processing.preprocessing import create_windows

WINDOW_SIZE = 4096
# Base case 50% overlap
STRIDE = 2048 

# Experiment
# STRIDE = 4096

# Experiment
# STRIDE = 1024

files = [
    ("Drone_Inspire", load_bin, "drone_rf/DJI_inspire_2_2G.bin"),
    ("Drone_Mavic", load_bin, "drone_rf/DJI_mavic_pro_2G.bin"),
    ("WiFi_30mbps", load_data, "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T132400M058863.data"),
    ("WiFi_50mbps", load_data, "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131300M012278.data"),
    ("LTE_30mbps", load_data, "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T150900M059227.data"),
    ("LTE_50mbps", load_data, "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152100M004125.data"),
    ("5G_NR", load_data, "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191200M056171.data"),
]

for name, loader, path in files:

    iq = loader(path)

    windows = create_windows(
        iq,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    assert windows.ndim == 2, f"{name}: Windows should be 2-dimensional."

    assert windows.shape[1] == WINDOW_SIZE, (
        f"{name}: Expected window size {WINDOW_SIZE}, got {windows.shape[1]}"
    )

    assert np.iscomplexobj(windows), (
        f"{name}: Windows should contain complex IQ samples."
    )

    assert len(windows) > 0, (
        f"{name}: No windows were generated."
    )

    print(f"✓ {name}")
    print(f"  Signal Shape : {iq.shape}")
    print(f"  Windows      : {windows.shape}")
    print(f"  Window Size  : {WINDOW_SIZE}")
    print(f"  Stride       : {STRIDE}")
    print()

print("✓ All window generation tests passed.")