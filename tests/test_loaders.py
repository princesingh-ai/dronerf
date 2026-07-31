from data_processing.loaders import load_bin, load_data
import numpy as np

DRONE_FILE = "/Users/prince/projects/dronerf/drone_rf/DJI_inspire_2_2G.bin"
NON_DRONE_FILE = "/Users/prince/projects/dronerf/random_rf/5G-NR/32274CF-dell-latitude-D20211201T191100M024240.data"

drone = load_bin(DRONE_FILE)
wifi = load_data(NON_DRONE_FILE)

assert np.iscomplexobj(drone)
assert np.iscomplexobj(wifi)

assert drone.ndim == 1
assert wifi.ndim == 1

assert len(drone) > 0
assert len(wifi) > 0

print("All loader tests passed")