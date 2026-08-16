import yaml
import torch
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Global Configuration Dictionary
config = load_config(CONFIG_PATH)

# Dynamic Device Selection (MPS for Apple Silicon, CUDA for NVIDIA, fallback to CPU)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
