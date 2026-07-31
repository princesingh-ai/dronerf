DroneRF Dataset & Signal Processing Manual
============================================

Data Acquisition, Physical Signal Theory, Ingestion, and Preprocessing Specifications

Overview & Dataset Rationale
----------------------------

Deep learning models trained on electromagnetic signals depend strictly on data
integrity, proper normalization, and leak-free validation splitting. Unlike optical
images or audio recordings, Radio Frequency (RF) signals captured by Software
Defined Radios (SDRs) represent high-rate complex-valued continuous wave arrays.

The DroneRF dataset pipeline provides structured routines to parse raw binary RF
recordings, perform recording-level train/validation/test splits, extract strided
sliding windows, and stream memory-mapped sample tensors during model training.

This technical manual details the underlying physical signal theory, dataset sources,
splitting strategy, windowing mechanics, disk schema, design decisions, and testing
protocols governing the dataset.


Purpose
-------

The purpose of the dataset subsystem in DroneRF is threefold:

1. Signal Physics Ingestion: Correctly extract 16-bit signed integer interleaved
   In-Phase ($I$) and Quadrature ($Q$) radio components from raw binary recordings
   and construct complex time-series representations ($I + jQ$).

2. Strict Data Leakage Prevention: Enforce dataset splitting at the original raw
   recording file level prior to sliding window extraction. This prevents adjacent
   overlapping windows from spilling across training and testing partitions.

3. Zero-Copy High-Throughput I/O: Enable memory-mapped lazy array indexing to allow
   training across large multi-gigabyte datasets without exceeding physical RAM.


Dataset Sources & External References
-------------------------------------

The primary dataset used for benchmark evaluation in this repository is derived from
the public DroneRF dataset collected by Allahham et al. (2019) at Oklahoma State University.

Dataset Overview & Link
~~~~~~~~~~~~~~~~~~~~~~~

* Dataset Name: DroneRF (Radio Frequency Dataset for Drone Detection)
* Principal Authors: M. S. Allahham, M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, A. Erbad,
  M. Guizani, and A. Khattab.
* Dataset Publication: "DroneRF dataset: A dataset of radio frequency signals for
  wireless autonomous vehicle detection," Data in Brief, vol. 26, p. 104301, 2019.
* Official DOI Link: https://doi.org/10.1016/j.dib.2019.104301
* Official Repository / Data Host: Mendeley Data (Version 1)
  Link: https://data.mendeley.com/datasets/zddmv59wvg/1

Hardware Ingestion Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw RF signals were captured using an RF receiver system consisting of two
SDR receiver channels connected to omnidirectional antennas:

* Primary Receiver: USRP B210 Software Defined Radio (SDR) by Ettus Research.
* Intermediate Frequency (IF) / RF Center Frequencies:
  - 2.4 GHz ISM Band (covering standard drone control and analog/digital video links).
  - 5.8 GHz ISM Band (covering high-bandwidth FPV video feeds).
* Receiver Sampling Rate: 40 MHz (40 Million Complex IQ Samples per Second).
* Bit Depth: 16-bit signed integers (`int16`), interleaved ($I_0, Q_0, I_1, Q_1, \dots$).

Recording Categories
~~~~~~~~~~~~~~~~~~~~

The dataset contains RF recordings organized into discrete transmission states:

1. Drone Classes (`drone_rf/`):
   - Active RF communications between commercial drones (e.g., DJI Phantom 4, Bebop 2,
     AR Drone) and ground controllers.
   - Includes drone turned on (RF connection established), drone hovering, drone flying,
     and active video streaming.

2. Non-Drone Classes (`random_rf/`):
   - Background electromagnetic activity in the 2.4 GHz and 5.8 GHz ISM bands without
     active drone transmissions.
   - Includes background RF noise, Wi-Fi traffic (IEEE 802.11a/b/g/n/ac), Bluetooth
     burst transmissions, and ambient radio interference.

Dataset License
~~~~~~~~~~~~~~~

The DroneRF dataset is distributed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).


Radio Frequency (RF) Signal Theory for CS Students
--------------------------------------------------

Radio frequency signals differ fundamentally from standard 1D scalar sequences
(such as audio waveforms) and 2D pixel grids (such as optical images). This section
explains the mathematical and physical foundations of RF signals from first principles.

What is a Continuous Radio Wave?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A radio signal is an electromagnetic wave transmitted through space at a specific
carrier frequency $f_c$ (for example, $f_c = 2.4 \text{ GHz} = 2.4 \times 10^9 \text{ cycles/second}$).

Mathematically, a pure sinusoidal radio wave is defined by three properties:

    s(t) = A(t) * cos( 2 * pi * f_c * t + phi(t) )

where:
* $A(t)$ is the instantaneous amplitude (signal strength).
* $f_c$ is the carrier frequency.
* $\phi(t)$ is the instantaneous phase (position within the sine wave cycle).

Information (such as drone steering commands or video frames) is embedded into the
wave by modulating $A(t)$ (Amplitude Modulation), $f_c$ (Frequency Modulation),
or $\phi(t)$ (Phase Modulation).

In-Phase (I) and Quadrature (Q) Signal Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~--------------------

Directly digitizing a 2.4 GHz carrier wave would require an Analog-to-Digital Converter
(ADC) sampling at over 4.8 GHz (according to the Nyquist-Shannon Sampling Theorem),
generating hundreds of gigabits of data per second.

To avoid this, Software Defined Radios perform down-conversion: they multiply the
incoming antenna signal by a local oscillator signal at frequency $f_c$, shifting the
high-frequency radio signal down to baseband ($0 \text{ Hz}$).

To capture both amplitude $A(t)$ and phase $\phi(t)$ without loss of information,
the receiver splits the signal into two orthogonal channels:

1. In-Phase Channel ($I$): The signal multiplied by $\cos(2\pi f_c t)$.
2. Quadrature Channel ($Q$): The signal multiplied by $-\sin(2\pi f_c t)$ (90-degree phase offset).

Mathematically, the baseband signal $x(t)$ is represented as a complex number:

    x(t) = I(t) + j * Q(t)

where:
* Magnitude $|x(t)| = \sqrt{ I(t)^2 + Q(t)^2 }$ represents the instantaneous signal envelope power.
* Phase $\arg(x(t)) = \arctan\left(\frac{Q(t)}{I(t)}\right)$ represents the instantaneous signal phase.

Signal-to-Noise Ratio (SNR) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In real-world deployment, received RF signals contain additive environmental thermal noise
and multipath reflections. The Signal-to-Noise Ratio (SNR) in decibels (dB) is defined as:

    SNR_dB = 10 * log10( P_signal / P_noise )

where $P_{\text{signal}} = \frac{1}{N} \sum |x_{\text{signal}}[i]|^2$ and $P_{\text{noise}} = \frac{1}{N} \sum |x_{\text{noise}}[i]|^2$.
High SNR values (>15 dB) represent clear line-of-sight drone signals, whereas low SNR values (<0 dB)
indicate weak signals attenuated by obstacles or distant operation.

IQ Constellation Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When plotted in a two-dimensional Cartesian plane where the X-axis represents In-Phase ($I$)
and the Y-axis represents Quadrature ($Q$), the sequence of samples forms an IQ Constellation
Diagram. 

For continuous wave signals, amplitude variations trace concentric circles or trajectories
around the origin. Drone control links utilizing Frequency Hopping Spread Spectrum (FHSS) or
Direct Sequence Spread Spectrum (DSSS) create distinct geometric constellation clusters
that set them apart from ambient thermal noise or Wi-Fi OFDM subcarrier patterns.

Raw Binary File Interleaving Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an SDR records complex signals to a disk file (`.bin` or `.data`), it writes
alternating 16-bit signed integers:

  Memory Byte Offset:  0x00  0x02  0x04  0x06  0x08  0x0A  0x0C  0x0E
                      +-----+-----+-----+-----+-----+-----+-----+-----+
  Sample Data:        | I0  | Q0  | I1  | Q1  | I2  | Q2  | I3  | Q3  |
                      +-----+-----+-----+-----+-----+-----+-----+-----+
  Data Type:          int16 int16 int16 int16 int16 int16 int16 int16

`data_processing/loaders.py` reads these bytes into Python memory using NumPy:

    raw_data = np.fromfile(path, dtype=np.int16)
    I = raw_data[0::2].astype(np.float32)
    Q = raw_data[1::2].astype(np.float32)
    iq_complex = I + 1j * Q


Dataset Split Strategy (Train / Validation / Test)
--------------------------------------------------

Dataset Split Ratios
~~~~~~~~~~~~~~~~~~~~

The dataset is partitioned into three discrete sub-datasets:
* Training Set: 80% of total raw recording files.
* Validation Set: 10% of total raw recording files.
* Test Set: 10% of total raw recording files.

Why Split BEFORE Preprocessing? (Preventing Data Leakage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A critical vulnerability in time-series machine learning is data leakage caused by
improper window splitting.

In DroneRF, long raw recordings are sliced into sliding windows of length 4096 with a
50% overlap (stride = 2048 samples). Consequently, Window $k$ and Window $k+1$ share
2048 identical time samples:

  Window k:   [ S_0,    S_1,  ..., S_2047 | S_2048, S_2049, ..., S_4095 ]
  Window k+1:                             [ S_2048, S_2049, ..., S_4095 | S_4096, ... ]
                                            \_________________________/
                                                 Shared 2048 Samples

If an engineer performs window extraction across all raw files *first*, and then
randomly shuffles all generated windows into train/validation/test sets:
* Window $k$ may end up in the Training Set.
* Window $k+1$ (sharing 50% identical samples with Window $k$) ends up in the Validation Set.

This results in severe Data Leakage: the neural network evaluates validation performance
on samples it has already seen during training, producing artificially near-100%
validation accuracy that completely fails when deployed on novel unseen RF recordings.

To guarantee true evaluation integrity, `data_processing/split.py` shuffles raw
*recording files* at the file path level using a fixed random seed (seed=42) *before*
any window slicing occurs:

  Raw Binary Files (.bin)
         │
         ▼
  Shuffle File Paths (split_dataset)
         │
         ├──> 80% Files ──> Create Windows ──> train/
         ├──> 10% Files ──> Create Windows ──> validation/
         └──> 10% Files ──> Create Windows ──> test/


Windowing & Sampling Parameters Rationale
----------------------------------------

Window Size: 4096 Samples
~~~~~~~~~~~~~~~~~~~~~~~~~

* Parameter: `window_size = 4096`
* Time Duration Rationale:
  At a typical SDR receiver sampling rate of 20 MHz ($20 \times 10^6$ samples/sec),
  a window of 4096 samples corresponds to a temporal duration of:

      T_window = 4096 / (20 * 10^6) = 0.0002048 seconds = 0.2048 milliseconds

  This duration is long enough to capture multiple complete symbol cycles of drone
  frequency-hopping telemetry bursts while remaining short enough to allow instantaneous
  model inference (<1 millisecond per window).

Window Overlap: 50% (Stride = 2048 Samples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Parameter: `stride = 2048` (50% overlap)
* Rationale:
  Transient RF transmissions (such as short control bursts or telemetry pings) may
  occur at the boundary of a fixed window. Without overlap, boundary events are
  cut in half, reducing model detection capability. A 50% stride guarantees that
  every transient burst appears un-truncated near the center of at least one sliding
  window, without inflating disk storage requirements excessively.


Normalization Strategy: Dynamic Peak Scaling
--------------------------------------------

Peak Normalization Formula
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each 4096-sample complex IQ window $x$ is normalized individually using peak scaling:

    peak = max( | x[i] | ) for i in 0..4095
    x_normalized = x / peak  (if peak > 0)

This maps the absolute magnitude of every sample into the range $[0.0, 1.0]$, and
replaces real and imaginary values into $[-1.0, 1.0]$.

Why Normalize During DataLoader Ingestion (Lazy Loading)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than saving normalized 32-bit floating point arrays directly to disk during
`create_dataset()`, window generation saves raw complex arrays, and peak normalization
is applied on-the-fly inside `RFDataset.__getitem__()` in `datasets/rf_dataset.py`.

Key Reasons for Lazy Normalization:

1. Disk Space Preservation: Raw complex arrays can be stored efficiently.
2. Signal Amplitude Integrity: Un-normalized raw data preserves absolute received
   signal strength indicators (RSSI) on disk, allowing future researchers to test
   alternative power-based normalization strategies without re-processing raw binary files.
3. Floating-Point Precision: Applying peak normalization dynamically in RAM prevents
   quantization degradation during serialization.


Directory Structure & Disk Schema
---------------------------------

When dataset creation executes via `create_dataset()`, files are structured in `processed/`:

    processed/
    ├── train/
    │   ├── drone/
    │   │   ├── recording_drone_01.npy
    │   │   └── recording_drone_02.npy
    │   └── non_drone/
    │       ├── recording_wifi_01.npy
    │       └── recording_bluetooth_01.npy
    ├── validation/
    │   ├── drone/
    │   └── non_drone/
    └── test/
        ├── drone/
        └── non_drone/

Each `.npy` file contains a 2D NumPy array of shape `(Num_Windows, 4096)` and data type
`np.complex64`.


Implementation
--------------

Binary Loader (`data_processing/loaders.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_bin(path) -> np.ndarray:
        raw_data = np.fromfile(path, dtype=np.int16)
        if len(raw_data) % 2 != 0:
            raise ValueError(f"{path} does not contain an even number of values.")
        I = raw_data[0::2].astype(np.float32)
        Q = raw_data[1::2].astype(np.float32)
        return I + 1j * Q

Sliding Window Generator (`data_processing/preprocessing.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def create_windows(iq: np.ndarray, window_size: int = 4096, stride: int | None = None):
        if stride is None:
            stride = window_size // 2
        windows = []
        for start in range(0, len(iq) - window_size + 1, stride):
            windows.append(iq[start : start + window_size])
        return np.array(windows)

Dataset Indexing (`datasets/rf_dataset.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    class RFDataset(Dataset):
        def __init__(self, dataset_path: str):
            self.samples = []
            dataset_path = Path(dataset_path)
            for file in (dataset_path / "drone").glob("*.npy"):
                windows = np.load(file, mmap_mode="r")
                for index in range(len(windows)):
                    self.samples.append((file, index, DRONE_LABEL))
            for file in (dataset_path / "non_drone").glob("*.npy"):
                windows = np.load(file, mmap_mode="r")
                for index in range(len(windows)):
                    self.samples.append((file, index, NON_DRONE_LABEL))

        def __getitem__(self, index):
            file, window_index, label = self.samples[index]
            windows = np.load(file, mmap_mode="r")
            window = windows[window_index]
            window = normalize(window)
            window = np.stack((window.real, window.imag), axis=0)
            return torch.from_numpy(window.astype(np.float32)), label


Design Decisions & Rationale
----------------------------

1. Why 80/10/10 Split Rationale?
   * Rationale: Allocating 80% of data to training provides sufficient diversity across
     drone flight modes, while 10% validation ensures reliable early stopping without
     over-fitting hyper-parameters. The isolated 10% test set provides an unbiased final
     benchmark score.

2. Why File-Level Splitting Rationale?
   * Rationale: Prevents catastrophic data leakage between 50% overlapping windows,
     ensuring real-world deployment reliability.

3. Why 4096 Window Size Rationale?
   * Rationale: Strikes an optimal balance between time-frequency resolution and low-latency
     inference (~0.2 ms capture duration).

4. Why Peak Normalization Rationale?
   * Rationale: Prevents high-power transmitters located close to antennas from dominating
     gradient updates, while scaling weak signals to comparable numerical ranges.


Testing & Verification
----------------------

The dataset pipeline is verified using automated `pytest` suites:

* `tests/test_loaders.py`: Verifies `load_bin()` correctly raises errors for odd-length
  binary files and properly de-interleaves real and imaginary components.
* `tests/test_preprocessing.py`: Tests `create_windows()` boundary conditions, ensuring
  partial windows shorter than 4096 are dropped and stride offsets are exact.
* `tests/test_split.py`: Asserts that set intersection of train, validation, and test
  file paths is strictly empty (zero file overlap).
* `tests/test_rf_dataset.py`: Verifies `RFDataset` length calculations and validates
  tensor output shape `(2, 4096)` and data type (`torch.float32`).

