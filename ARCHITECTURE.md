DroneRF Architecture & System Specification
==========================================

Technical System Design, Module Specifications, and Data Pipelines

Overview & Design Philosophy
----------------------------

DroneRF is built around the principles of high-performance signal ingestion,
strict memory efficiency, separation of concerns, and mathematical numerical
stability. The project processes large-scale high-rate Radio Frequency (RF)
recordings containing raw complex In-Phase and Quadrature (IQ) samples.

Because raw binary RF recordings routinely exceed available RAM when scaled across
multi-hour operational captures, the architecture separates data preparation
(file parsing, dataset splitting, sliding window slicing) from model ingestion
(lazy memory mapping, dynamic peak scaling, and batched DataLoader streaming).

This document details every architectural module, tensor transformation, control
flow sequence, and neural network block within the DroneRF system.


Purpose
-------

The architectural goal of DroneRF is to establish a robust, modular, and
production-grade software framework for deep-learning-based RF classification.

Specific architectural objectives include:

1. Zero-Leakage Dataset Splitting: Guaranteeing that temporal window extraction
   never leaks adjacent sliding windows across training, validation, and testing.

2. Zero-Copy RAM Streaming: Leveraging memory-mapped file handles to access multi-gigabyte
   array datasets without incurring RAM overflow or slow disk deserialization.

3. Low-Latency 1D Inference: Structuring a compact 1D Convolutional Neural Network
   (DroneCNN) capable of running end-to-end inference directly on 2-channel complex
   time series tensors.

4. Resilient Checkpoint Management: Preserving training momentum and model weights
   atomically during validation loss improvements.


System Top-Level Architecture
-----------------------------

The complete software environment is organized into five isolated architectural
layers: Data Ingestion Layer, Preprocessing & Disk Layer, Dataset & Loader Layer,
Neural Network Layer, and Training & Checkpoint Engine.

  +-------------------------------------------------------------------------+
  |                          DATA INGESTION LAYER                           |
  |  load_bin() / load_data() -> Reads raw int16 interleaved IQ from disk   |
  +------------------------------------v------------------------------------+
                                       |
  +------------------------------------v------------------------------------+
  |                    PREPROCESSING & DISK LAYER                           |
  |  split_dataset() -> Recording-level 80/10/10 random file assignment     |
  |  create_windows() -> Slices signals into 4096-sample windows (50% stride)|
  |  process_split()  -> Saves contiguous windows as NumPy .npy files       |
  +------------------------------------v------------------------------------+
                                       |
  +------------------------------------v------------------------------------+
  |                    DATASET & LOADER LAYER                               |
  |  RFDataset        -> Lazy mmap read, peak normalize, complex to 2xN    |
  |  DataLoader       -> Batched parallel PyTorch tensor generation         |
  +------------------------------------v------------------------------------+
                                       |
  +------------------------------------v------------------------------------+
  |                    NEURAL NETWORK LAYER                                 |
  |  DroneCNN         -> 3-Stage 1D Conv + Adaptive Pool + Linear Logit     |
  +------------------------------------v------------------------------------+
                                       |
  +------------------------------------v------------------------------------+
  |                TRAINING & CHECKPOINT ENGINE                             |
  |  train_one_epoch  -> Forward pass, loss calculation, backpropagation    |
  |  validate         -> Evaluation mode forward pass & loss aggregation     |
  |  save_checkpoint  -> Atomic checkpoint persistence to disk (.pt file)   |
  +-------------------------------------------------------------------------+


Module Breakdown & Responsibilities
----------------------------------

1. Package: `data_processing`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `data_processing` package is responsible for all offline data transformations
before dataset loading.

* `data_processing/loaders.py`:
  - `load_bin(path)`: Opens binary files containing 16-bit signed integers
    (`np.int16`), validates that the sample byte length is even, extracts alternating
    In-Phase (even indices) and Quadrature (odd indices) components, converts them
    to 32-bit floats (`np.float32`), and combines them into complex64 values
    (`I + 1j * Q`).
  - `load_data(path)`: Provides parsing for secondary `.data` raw binary formats.

* `data_processing/split.py`:
  - `split(files, train_ratio, val_ratio)`: Segregates a list of file paths into
    three sub-lists based on proportional index cutoffs.
  - `split_dataset(...)`: Accepts drone file paths and non-drone file paths, validates
    that ratios sum to 1.0 (80% train, 10% validation, 10% test), sets a fixed random
    seed (seed=42) for deterministic reproduction, shuffles file lists independently,
    and returns six discrete file arrays.

* `data_processing/preprocessing.py`:
  - `normalize(iq)`: Computes maximum absolute magnitude `peek = np.max(np.abs(iq))`
    and divides the array by `peek`. Returns un-modified signal if peak is zero.
  - `create_windows(iq, window_size=4096, stride=2048)`: Iterates across the complex
    signal using a fixed stride, slicing sub-arrays of length 4096. Assembles the
    sliced arrays into a contiguous 2D NumPy array of shape `(Num_Windows, 4096)`.

* `data_processing/save.py`:
  - `process_split(files, output_dir)`: Iterates over a list of raw recording paths,
    invokes `load_bin()`, calls `create_windows()`, and saves the window array into
    `output_dir` as a `.npy` file using `np.save()`.
  - `create_dataset(drone_path, non_drone_path, output_dir)`: Coordinates full
    dataset creation by discovering `.bin` and `.data` files, calling `split_dataset()`,
    and executing `process_split()` across all six dataset partitions.

2. Package: `datasets`
~~~~~~~~~~~~~~~~~~~~~~

* `datasets/rf_dataset.py`:
  - Defines `RFDataset(torch.utils.data.Dataset)`.
  - In `__init__`, scans `drone/` and `non_drone/` directories for `.npy` files.
    Opens each `.npy` file using `np.load(file, mmap_mode="r")` without reading data
    into RAM, builds an index list containing tuples of `(file_path, window_index, label)`.
  - In `__getitem__(index)`, reads the exact 4096-sample complex array from disk,
    applies peak normalization, splits complex numbers into real (`I`) and imaginary
    (`Q`) components, stacks them into a 2D float32 array of shape `(2, 4096)`, and
    converts it to a PyTorch tensor.

3. Package: `models`
~~~~~~~~~~~~~~~~~~~~

* `models/cnn1d.py`:
  - Implements `DroneCNN(nn.Module)`.
  - Contains `self.features` (Sequential 1D convolutions, batch normalization,
    ReLU activations, max pooling, adaptive average pooling) and `self.classifier`
    (Flattening and single Linear output neuron).

4. Package: `training`
~~~~~~~~~~~~~~~~~~~~~~

* `training/config.py`: Global hyperparameter store (`BATCH_SIZE=64`,
  `LEARNING_RATE=0.001`, `EPOCHS=5`, `NUM_WORKERS=0`, `DEVICE` auto-detection).
* `training/data_loader.py`: `create_dataloaders()` instantiates `RFDataset` for
  `train`, `validation`, and `test` splits and wraps them in PyTorch `DataLoader` instances.
* `training/train.py`: `train_one_epoch()` executes model forward passes, calculates
  `BCEWithLogitsLoss`, zeroing gradients, backpropagating loss, and updating weights.
* `training/validate.py`: `validate()` runs model evaluation without gradient computation
  (`torch.no_grad()`) and aggregates validation loss.
* `training/metrics.py`: Computes thresholded binary classification performance:
  `accuracy()`, `precision()`, `recall()`, and `f1_score()`.
* `training/check_points.py`: `save_checkpoint()` and `load_checkpoint()` manage atomic
  serialization of model parameters, optimizer states, epoch numbers, and loss values.


Data Flow & Signal Representation
---------------------------------

The transformations undergone by an RF recording from raw disk bytes to neural output
are illustrated below:

Step 1: Raw Int16 Interleaved File (.bin)
-----------------------------------------
File Bytes: [ I0_LSB, I0_MSB, Q0_LSB, Q0_MSB, I1_LSB, I1_MSB, Q1_LSB, Q1_MSB, ... ]
Parsed Array (np.int16): [ I0, Q0, I1, Q1, I2, Q2, ... ]

Step 2: Complex Signal Assembly (loaders.py)
--------------------------------------------
De-interleaved Arrays (float32):
  I_array = [ I0, I1, I2, I3, ... ]
  Q_array = [ Q0, Q1, Q2, Q3, ... ]
Complex Array (complex64):
  IQ = I_array + 1j * Q_array  -->  [ (I0 + jQ0), (I1 + jQ1), (I2 + jQ2), ... ]

Step 3: Sliding Window Slicing (preprocessing.py)
------------------------------------------------
Given IQ of length L = 100,000, window_size = 4096, stride = 2048:
  Window 0: IQ[0 : 4096]
  Window 1: IQ[2048 : 6144]
  Window 2: IQ[4096 : 8192]
  ...
Shape saved to disk (.npy): (Num_Windows, 4096) [dtype: complex64]

Step 4: PyTorch Dataset Feature Extraction (rf_dataset.py)
----------------------------------------------------------
For single window index:
  Complex Window (4096,): [ I0+jQ0, I1+jQ1, ..., I4095+jQ4095 ]
  Peak Normalization: peak = max(abs(window)); window = window / peak
  Channel Stacking:
    Row 0 (I channel): [ I0, I1, I2, ..., I4095 ]
    Row 1 (Q channel): [ Q0, Q1, Q2, ..., Q4095 ]
  Tensor Output: Shape (2, 4096) [dtype: torch.float32]

Step 5: DataLoader Batch Assembly (data_loader.py)
--------------------------------------------------
Batch Tensor: Shape (Batch_Size, 2, 4096)
Label Tensor: Shape (Batch_Size, 1)

Step 6: Neural Network Forward Pass (cnn1d.py)
----------------------------------------------
Output Logit Tensor: Shape (Batch_Size, 1)


Control Flow & Pipeline State Machine
-------------------------------------

The runtime operational execution flow of `main.py` is governed by a synchronous
training loop:

                      +-----------------------------+
                      |     Initialize Config       |
                      | (Batch, LR, Epochs, Device) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Instantiate DataLoaders   |
                      |  (Train, Validation, Test)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Instantiate DroneCNN    |
                      |    Move model to DEVICE     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Instantiate Optimizer    |
                      | (Adam) & Criterion (BCE)    |
                      +--------------+--------------+
                                     |
                                     v
            +------------> Epoch Loop (1..EPOCHS)
            |                        |
            |                        v
            |         +-----------------------------+
            |         |     train_one_epoch()       |
            |         | Set model.train(), ZeroGrad |
            |         | Forward pass, Backward pass |
            |         |  Optimizer weight update    |
            |         +--------------+--------------+
            |                        |
            |                        v
            |         +-----------------------------+
            |         |         validate()          |
            |         | Set model.eval(), NoGrad    |
            |         |  Compute validation loss    |
            |         +--------------+--------------+
            |                        |
            |                        v
            |         +-----------------------------+
            |         |  Is Val Loss < Best Loss?   |
            |         +--------------+--------------+
            |                        |
            |             +----------+----------+
            |             | YES                 | NO
            |             v                     v
            |  +---------------------+   (Continue to
            |  |  save_checkpoint()  |    next epoch)
            |  |  Update best loss   |          |
            |  +----------+----------+          |
            |             |                     |
            +-------------+---------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Training Complete       |
                      +-----------------------------+


Deep Neural Network Architecture (DroneCNN)
-------------------------------------------

`DroneCNN` is a 1D Convolutional Neural Network engineered specifically for feature
extraction from 2-channel temporal RF sequences.

Theoretical Foundation of 1D Convolution on Complex IQ Signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 1D convolutional filter operating on a 2-channel input (In-phase $I[t]$ and
Quadrature $Q[t]$) computes a spatial-temporal cross-correlation. For kernel
weights $w_I$ and $w_Q$ of size $K$, the feature map output at temporal position
$t$ is defined mathematically as:

    y[t] = sum_{k=0}^{K-1} ( w_I[k] * I[t+k] + w_Q[k] * Q[t+k] ) + b

By learning joint linear combinations across both $I$ and $Q$ channels simultaneously,
1D convolutional kernels act as adaptive bandpass filters, phase discriminators, and
amplitude modulation detectors. This allows the network to automatically extract carrier
frequency shifts, symbol transition boundaries, and power spectral signatures directly
from raw time-series samples without human feature engineering.

Architectural Layer Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Convolutional Block 1:
   - Input Channels: 2 (I and Q)
   - Output Channels: 32
   - Kernel Size: 7 (Receptive field = 7 samples)
   - Padding: 3 (Preserves temporal length = 4096)
   - Stride: 1
   - Batch Normalization: `BatchNorm1d(32)`
   - Activation: `ReLU()`
   - Max Pooling: `MaxPool1d(kernel_size=2, stride=2)` (Halves temporal length to 2048)

2. Convolutional Block 2:
   - Input Channels: 32
   - Output Channels: 64
   - Kernel Size: 5 (Receptive field = 5 samples)
   - Padding: 2 (Preserves temporal length = 2048)
   - Stride: 1
   - Batch Normalization: `BatchNorm1d(64)`
   - Activation: `ReLU()`
   - Max Pooling: `MaxPool1d(kernel_size=2, stride=2)` (Halves temporal length to 1024)

3. Convolutional Block 3:
   - Input Channels: 64
   - Output Channels: 128
   - Kernel Size: 3 (Receptive field = 3 samples)
   - Padding: 1 (Preserves temporal length = 1024)
   - Stride: 1
   - Batch Normalization: `BatchNorm1d(128)`
   - Activation: `ReLU()`
   - Max Pooling: `MaxPool1d(kernel_size=2, stride=2)` (Halves temporal length to 512)

4. Classifier Head:
   - Global Pooling: `AdaptiveAvgPool1d(1)` (Collapses temporal length from 512 to 1)
   - Flattening: `nn.Flatten()` (Reshapes tensor from `(Batch, 128, 1)` to `(Batch, 128)`)
   - Linear Output Layer: `nn.Linear(in_features=128, out_features=1)`

Tensor Shape & Parameter Summary Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------+---------------------+-------------------+-------------------+
| Layer Name            | Input Shape         | Output Shape      | Parameter Count   |
+-----------------------+---------------------+-------------------+-------------------+
| Input Tensor          | (Batch, 2, 4096)    | (Batch, 2, 4096)  | 0                 |
| Conv1d Block 1        | (Batch, 2, 4096)    | (Batch, 32, 4096) | 2 * 7 * 32 + 32   |
| BatchNorm1d (32)      | (Batch, 32, 4096)   | (Batch, 32, 4096) | 32 * 2            |
| MaxPool1d (2)         | (Batch, 32, 4096)   | (Batch, 32, 2048) | 0                 |
| Conv1d Block 2        | (Batch, 32, 2048)   | (Batch, 64, 2048) | 32 * 5 * 64 + 64  |
| BatchNorm1d (64)      | (Batch, 64, 2048)   | (Batch, 64, 2048) | 64 * 2            |
| MaxPool1d (2)         | (Batch, 64, 2048)   | (Batch, 64, 1024) | 0                 |
| Conv1d Block 3        | (Batch, 64, 1024)   | (Batch, 128, 1024)| 64 * 3 * 128 + 128|
| BatchNorm1d (128)     | (Batch, 128, 1024)  | (Batch, 128, 1024)| 128 * 2           |
| MaxPool1d (2)         | (Batch, 128, 1024)  | (Batch, 128, 512) | 0                 |
| AdaptiveAvgPool1d (1) | (Batch, 128, 512)   | (Batch, 128, 1)   | 0                 |
| Flatten               | (Batch, 128, 1)     | (Batch, 128)      | 0                 |
| Linear (128 -> 1)     | (Batch, 128)        | (Batch, 1)        | 128 * 1 + 1       |
+-----------------------+---------------------+-------------------+-------------------+


Memory-Mapped Streaming Architecture
-----------------------------------

Traditional PyTorch dataset loading reads complete data files into system RAM during
Dataset initialization. For multi-gigabyte RF recordings, this causes system memory
exhaustion.

DroneRF addresses this through NumPy read-only memory mapping (`mmap_mode="r"`):

1. File Pointer Indexing: During `RFDataset.__init__`, files are opened with `mmap_mode="r"`.
   NumPy inspects the header of each `.npy` file on disk to determine sample shape
   `(Num_Windows, 4096)` without loading array data.

2. Operating System Page Table Management: Memory mapping creates virtual memory address
   pointers mapped directly to the storage subsystem. When PyTorch DataLoader worker
   threads request sample index $i$, the operating system issues virtual page faults only
   for the specific bytes containing window $i$.

3. Index Mapping Array: The dataset constructs an internal Python index list:

       self.samples = [
           (Path("processed/train/drone/file1.npy"), 0, Label=1),
           (Path("processed/train/drone/file1.npy"), 1, Label=1),
           ...
           (Path("processed/train/non_drone/file2.npy"), 0, Label=0),
       ]

4. On-Demand Page Fault Reads: When `__getitem__(idx)` is invoked by worker threads
   in `DataLoader`, operating system page faults read only the specific 4096-sample
   block from disk into RAM cache. Once normalized and batched into PyTorch tensors,
   unreferenced memory pages are reclaimed automatically by operating system cache
   eviction.


Training Infrastructure & Checkpoint Engine
-------------------------------------------

Checkpoint Serialization Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model checkpoints are stored using PyTorch binary format (`.pt`) containing a
dictionary of state representations:

    {
        "epoch": 4,
        "model_state_dict": OrderedDict(...),
        "optimizer_state_dict": { ... },
        "loss": 0.0412
    }

Atomic Checkpoint Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~

To prevent file corruption if training is interrupted during a disk write operation,
`save_checkpoint()` in `training/check_points.py` ensures directory creation before
serialization:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict, path)


Implementation
--------------

Modular Package Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~

The architecture strictly separates data handling (`data_processing`, `datasets`),
modeling (`models`), and execution (`training`, `main.py`).

Code Decoupling Benefits
~~~~~~~~~~~~~~~~~~~~~~~~

1. Preprocessing can be run independently on high-memory data preparation servers.
2. The dataset module (`rf_dataset.py`) remains completely agnostic to model architecture.
3. The neural network architecture (`cnn1d.py`) can be swapped with alternative models
   (e.g., ResNet-1D) without altering data loaders or training loops.


Design Decisions & Rationale
----------------------------

1. Why Modular Directory Architecture?
   * Rationale: Separating preprocessing, dataset loading, network architecture, and
     training loops allows independent testing, reuse of components, and maintainability.
     A monolithic script makes unit testing impossible and obscures code duties.

2. Why Decreasing Kernel Sizes (7 -> 5 -> 3)?
   * Rationale: The initial layer with kernel size 7 provides a large receptive field
     capable of capturing broad temporal carrier modulations across raw IQ samples.
     Subsequent layers with smaller kernels (5 and 3) refine increasingly localized,
     high-level spectral patterns while keeping computational complexity low.

3. Why Adaptive Average Pooling (`AdaptiveAvgPool1d(1)`)?
   * Rationale: Standard fully connected layers require exact, static input dimensions.
     If the input sequence length changes, standard fully connected layers fail.
     Global adaptive pooling aggregates spatial/temporal features across the entire
     sequence into a fixed 128-element feature vector, making the classifier head
     dimensionally robust and dramatically reducing parameter count to prevent overfitting.

4. Why Unscaled Logits with BCEWithLogitsLoss?
   * Rationale: Combining a Sigmoid activation function and Binary Cross-Entropy in a
     single class provides superior numerical stability. PyTorch implements the
     log-sum-exp trick internally, preventing exponent overflow when raw model logits
     exceed extreme bounds.

5. Why Memory-Mapped Data Access (`mmap_mode="r"`)?
   * Rationale: Allows training on datasets larger than physical RAM. It eliminates
     startup delays associated with dataset loading and minimizes memory footprint.


Testing & Verification Strategy
-------------------------------

The architectural integrity of DroneRF is continuously validated using `pytest`:

1. Unit Testing Module Contracts:
   * `tests/test_cnn.py`: Verifies forward pass execution of `DroneCNN` with synthetic
     tensors of shape `(Batch, 2, 4096)` and asserts output shape is strictly `(Batch, 1)`.
   * `tests/test_rf_dataset.py`: Mocks file paths and verifies that `__getitem__` returns
     normalized 2D tensors of correct shape and data type (`torch.float32`).
   * `tests/test_checkpoints.py`: Saves a dummy model state, loads it back, and asserts
     exact equality of state dictionary tensors.

2. End-to-End System Tests:
   * `tests/test_train.py` & `tests/test_validate.py`: Verifies that single training and
     validation iterations complete without throwing device mismatch or shape runtime errors.

