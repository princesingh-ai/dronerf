import torch

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
NUM_WORKERS = 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = "checkpoints/best_model.pt"
DATASET_PATH = "processed"