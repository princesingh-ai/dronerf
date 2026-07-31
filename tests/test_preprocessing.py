import numpy as np
from data_processing.loaders import load_bin, load_data
from data_processing.preprocessing import normalize


files = [
    ("Drone", load_bin, "drone_rf/DJI_inspire_2_2G.bin"),
    ("Drone", load_bin, "drone_rf/DJI_mavic_pro_2G.bin"),
    ("WiFi_30mbps", load_data, "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T132400M058863.data"),
    ("WiFi_50mbps", load_data, "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131300M012278.data"),
    ("LTE_50mbps", load_data, "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152100M004125.data"),
    ("LTE_30mbps", load_data, "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T150900M059227.data"),
    ("5G", load_data, "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191200M056171.data"),]

for label, loader, path in files:
    iq = loader(path)
    norm = normalize(iq)
    assert np.max(np.abs(norm)) <= 1.0 + 1e-6
    print(f"✓ {label} passed")