from data_processing.loaders import load_bin, load_data
import numpy as np

files = [
    ("Drone_Inspire", load_bin, "drone_rf/DJI_inspire_2_2G.bin"),
    ("Drone_Mavic", load_bin, "drone_rf/DJI_mavic_pro_2G.bin"),
    ("WiFi_30mbps", load_data, "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T132400M058863.data"),
    ("WiFi_50mbps", load_data, "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131300M012278.data"),
    ("LTE_50mbps", load_data, "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152100M004125.data"),
    ("LTE_30mbps", load_data, "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T150900M059227.data"),
    ("5G", load_data, "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191200M056171.data"),
]

for name, loader, path in files:

    signal = loader(path)

    assert np.iscomplexobj(signal), f"{name}: Output is not complex."
    assert signal.ndim == 1, f"{name}: Expected a 1D array."
    assert len(signal) > 0, f"{name}: Signal is empty."

    print(f"✓ {name}")
    print(f"  Shape : {signal.shape}")
    print(f"  Dtype : {signal.dtype}")
    print(f"  First : {signal[1000:1005]}")
    print()

print("✓ All loader tests passed")