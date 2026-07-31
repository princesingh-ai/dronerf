DroneRF Training, Loss Formulation & Optimization Manual
==========================================================

Model Optimization Dynamics, Loss Functions, Metrics, and Checkpointing

Overview & Training Philosophy
------------------------------

Model training in DroneRF translates raw complex time-series features into reliable
binary classification probabilities. The training infrastructure is designed for high
numerical stability, reproducibility, and cross-platform hardware acceleration.

By combining an un-activated 1D Convolutional Neural Network output logit with
PyTorch's `BCEWithLogitsLoss` loss function, the system avoids numerical underflow
and overflow during gradient backpropagation.

This manual details the complete optimization loop, loss function mathematics,
metric calculations, hyperparameter choices, checkpointing mechanics, device selection
logic, testing procedures, and design rationale.


Purpose
-------

The purpose of the training subsystem in DroneRF is to:

1. Optimize Network Weights: Minimize binary cross-entropy loss over 4096-sample
   IQ windows using the Adam optimizer.

2. Ensure Numerical Stability: Utilize logit-level loss evaluation to prevent gradient
   vanishing/explosion.

3. Track Operational Metrics: Calculate precision, recall, accuracy, and F1-score
   to evaluate real-world drone detection capability.

4. Manage State Serialization: Save atomic checkpoints whenever validation loss improves
   to prevent loss of training progress.


Training Pipeline Architecture
------------------------------

The training loop follows a structured sequence of forward passes, loss calculations,
gradient updates, validation evaluations, and checkpoint triggers.

                     +---------------------------------------+
                     |         Start Training Epoch          |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    Batch Ingestion (DataLoader)       |
                     |  Tensors: (64, 2, 4096), Labels: (64)  |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         Device Transfer               |
                     |  Move tensors to CUDA / MPS / CPU     |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         Optimizer ZeroGrad            |
                     |     Reset parameter gradients         |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        DroneCNN Forward Pass          |
                     |  Compute unscaled logits: (64, 1)     |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |     BCEWithLogitsLoss Evaluation      |
                     |  Compute binary cross-entropy loss    |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |       Gradient Backpropagation        |
                     |           loss.backward()             |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        Optimizer Step                 |
                     |          optimizer.step()             |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |       Validation Evaluation           |
                     |   Compute validation loss & metrics   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |   Is Val Loss < Best Val Loss?        |
                     +-------------------+-------------------+
                                         |
                               +---------+---------+
                               | YES               | NO
                               v                   v
                    +--------------------+   (Next Epoch)
                    |  save_checkpoint() |
                    +--------------------+


Loss Function Analysis: BCEWithLogitsLoss
----------------------------------------

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The loss function used for training is PyTorch's `nn.BCEWithLogitsLoss`. Given an
un-activated network output logit $x \in \mathbb{R}$ and a ground-truth binary label
$y \in \{0, 1\}$, the loss for a single sample is defined as:

    L(x, y) = - [ y * log( sigma(x) ) + (1 - y) * log( 1 - sigma(x) ) ]

where $\sigma(x)$ is the standard Sigmoid activation function:

    sigma(x) = 1 / ( 1 + exp(-x) )

Why Combine Sigmoid and BCE into Logit-Space Loss?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a model explicitly outputs $\hat{y} = \sigma(x)$ using an internal `nn.Sigmoid()`
layer, and passes $\hat{y}$ into standard `nn.BCELoss()`, the loss equation evaluates:

    L = - [ y * log( y_hat ) + (1 - y) * log( 1 - y_hat ) ]

When the network produces extreme output logits (for example, $x = -80$ or $x = +80$):
* $\sigma(-80)$ evaluates to $0.0$ in 32-bit floating point precision due to underflow.
* $\log(0.0)$ evaluates to $-\infty$ or `NaN` (Not a Number), causing immediate gradient
  corruption and terminating training.

`BCEWithLogitsLoss` prevents this instability by reformulating the loss equation into
a single mathematically equivalent expression using the Log-Sum-Exp trick:

    L(x, y) = max(x, 0) - x * y + log( 1 + exp( -|x| ) )

By isolating the absolute value $|x|$ inside $\exp(-|x|)$, the exponential term is
guaranteed to remain in the range $(0, 1]$, completely eliminating `NaN` overflow and
floating-point underflow.

Gradient Derivation of Logit Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The partial derivative of $\mathcal{L}(x, y)$ with respect to the raw logit $x$ simplifies
to an remarkably clean expression:

    dL / dx = sigma(x) - y

This gradient has two critical properties:
1. When $x$ is well-calibrated ($\sigma(x) \approx y$), the gradient approaches zero smoothly.
2. When the model makes an error (e.g., $y=1$ but $x$ is negative), the gradient magnitude
   $|\sigma(x) - y|$ scales proportionally to the error magnitude, producing bounded,
   stable gradient updates without exploding gradients.


Optimizer Selection: Adam Optimizer
-----------------------------------

Algorithmic Details
~~~~~~~~~~~~~~~~~~~

The model parameters are updated using the Adam (Adaptive Moment Estimation) optimizer.
Adam computes individual adaptive learning rates for each parameter by maintaining
exponential moving averages of both the first moment (mean gradients) and second moment
(uncentered variance of gradients).

Given gradient $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ at timestep $t$:

1. First Moment Update:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

2. Second Moment Update:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

3. Bias-Corrected Estimators:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4. Parameter Weight Update:
   $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Batch Normalization Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside `DroneCNN`, each 1D convolution layer is followed by `BatchNorm1d`. During training
(`model.train()`), batch normalization computes batch mean $\mu_B$ and variance $\sigma_B^2$
across the temporal sequence for each channel $c$:

    x_hat_{i, c, t} = ( x_{i, c, t} - mu_{B, c} ) / sqrt( sigma_{B, c}^2 + eps )
    y_{i, c, t} = gamma_c * x_hat_{i, c, t} + beta_c

Batch normalization smooths the loss landscape, reduces internal covariate shift, and allows
the Adam optimizer to operate safely at a higher learning rate ($\alpha = 10^{-3}$).

Hyperparameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Learning Rate ($\alpha$): $1 \times 10^{-3}$ ($0.001$).
* Exponential Decay Rate 1 ($\beta_1$): $0.9$.
* Exponential Decay Rate 2 ($\beta_2$): $0.999$.
* Numerical Epsilon ($\epsilon$): $1 \times 10^{-8}$.

Why Adam over SGD for Raw RF Signals?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw complex RF signals contain transient bursts of high power interspersed with low-power
background noise. Standard Stochastic Gradient Descent (SGD) with a fixed learning rate
struggles under varying gradient magnitudes across sparse spectral events. Adam's adaptive
per-parameter scaling accelerates convergence and prevents gradient stall during initial
epochs.


Evaluation Metrics & Formulations
---------------------------------

While binary loss directs parameter optimization, model performance is evaluated using
four classification metrics implemented in `training/metrics.py`.

Thresholding Rule
~~~~~~~~~~~~~~~~~

Because `DroneCNN` outputs unscaled logits $x$, predictions are thresholded at $x \ge 0.0$
(which corresponds to $\sigma(x) \ge 0.5$):

    y_pred = 1  if x >= 0.0  else  0

Metric Formulations
~~~~~~~~~~~~~~~~~~~

1. Accuracy:
   Ratio of correct predictions over total predictions:

       Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. Precision:
   Ratio of true drone detections over all predicted drone detections:

       Precision = TP / (TP + FP)

3. Recall (Sensitivity):
   Ratio of true drone detections over all actual drone transmissions:

       Recall = TP / (TP + FN)

4. F1-Score:
   Harmonic mean of Precision and Recall:

       F1 = 2 * (Precision * Recall) / (Precision + Recall)

Why Recall is Critical for RF Drone Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In perimeter security applications, a False Negative ($FN$, failing to detect an active
drone) represents an un-flagged security threat, whereas a False Positive ($FP$, flagging
background Wi-Fi as a drone) is merely a temporary alarm audit. High Recall is therefore
prioritized over absolute raw accuracy.


Validation & Checkpointing Subsystem
------------------------------------

Validation Engine (`training/validate.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the end of each training epoch, the system executes `validate()`. The model is placed
in evaluation mode (`model.eval()`), disabling batch normalization statistic updates.
Gradients are turned off (`with torch.no_grad()`) to conserve GPU memory and execution speed.

Validation loss is averaged over all validation loader batches:

    validation_loss = total_running_loss / len(validation_loader)

Atomic Checkpoint Saving (`training/check_points.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If `validation_loss < best_validation_loss`:
1. `best_validation_loss` is updated.
2. `save_checkpoint()` serializes the state to `checkpoints/best_model.pt`:

       checkpoint = {
           "epoch": epoch + 1,
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
           "loss": validation_loss,
       }
       torch.save(checkpoint, "checkpoints/best_model.pt")

Resuming Training Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~

To resume an interrupted training run or fine-tune from a saved model checkpoint:

    from training.check_points import load_checkpoint
    from models.cnn1d import DroneCNN
    import torch

    model = DroneCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, optimizer, start_epoch, best_loss = load_checkpoint(
        model, optimizer, "checkpoints/best_model.pt"
    )


Hyperparameters & Device Auto-Detection
---------------------------------------

Hyperparameter Table (`training/config.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+-----------------------+---------------------------------------+
| Parameter Name      | Config Value          | Architectural Rationale               |
+---------------------+-----------------------+---------------------------------------+
| BATCH_SIZE          | 64                    | Balances GPU memory and SGD noise     |
| LEARNING_RATE       | 0.001 (1e-3)          | Standard Adam initial learning rate   |
| EPOCHS              | 5                     | Initial baseline convergence epoch count|
| NUM_WORKERS         | 0                     | Prevents inter-process mmap lock delays|
| MODEL_PATH          | checkpoints/best_model.pt | Output binary checkpoint path     |
| DATASET_PATH        | processed             | Disk directory containing window sets |
+---------------------+-----------------------+---------------------------------------+

Device Selection Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~

`training/config.py` automatically discovers available hardware accelerators:

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

1. `cuda`: Enables NVIDIA GPU hardware acceleration via CUDA drivers.
2. `mps`: Enables Apple Silicon GPU hardware acceleration via Metal Performance Shaders.
3. `cpu`: Fallback multithreaded CPU computation.


Implementation
--------------

Training Iteration (`training/train.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def train_one_epoch(model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for windows, labels in dataloader:
            windows = windows.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(windows)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(dataloader)


Design Decisions & Rationale
----------------------------

1. Why BCEWithLogitsLoss over Sigmoid + BCELoss?
   * Rationale: Log-Sum-Exp integration prevents floating-point underflow/overflow when
     logits reach extreme bounds ($|x| > 80$), preserving training stability.

2. Why Adam Optimizer over SGD?
   * Rationale: Adaptive per-parameter learning rates handle non-stationary RF power
     bursts and sparse spectral activations effectively.

3. Why Checkpoint Triggered on Validation Loss?
   * Rationale: Training loss continuously decreases even when a model begins to overfit.
     Saving checkpoints exclusively when validation loss reaches a new minimum ensures
     optimal generalization performance on novel unseen signals.


Testing & Verification
----------------------

Training stability is tested using `pytest`:

* `tests/test_train.py`: Mocks a synthetic dataset and DataLoader, executes
  `train_one_epoch()`, and asserts that model weights change (verifying non-zero gradients).
* `tests/test_validate.py`: Asserts that `validate()` returns a valid float loss without
  mutating model parameter tensors.
* `tests/test_metrics.py`: Tests `accuracy()`, `precision()`, `recall()`, and `f1_score()`
  against known edge-case target tensors (e.g., all positive, all negative, 50% split).
* `tests/test_checkpoints.py`: Saves model weights to a temporary file, mutates model
  weights, loads the checkpoint, and asserts exact numerical state restoration.

