import json
import numpy as np
import torch
from collections import Counter
from pathlib import Path

from data_processing.preprocessing import create_windows, normalize
from models.cnn1d_multiclass import MultiClassDroneCNN
from training.check_points import load_checkpoint
from utils.config import DEVICE, config

class DroneInference:
    """
    Inference wrapper around the MultiClassDroneCNN model.
    Transforms complex IQ -> 4096-sample windows -> normalized [B, 2, 4096] -> argmax multi-class prediction.
    """

    def __init__(self, model_path: str = None, batch_size: int = 128, mapping_path: str = None):
        if model_path is None:
            model_path = config["training"]["model_path"]
        if mapping_path is None:
            mapping_path = config["dataset"]["mapping_path"]
        self.device = DEVICE
        self.batch_size = batch_size
        
        # Load class mapping
        if not Path(mapping_path).exists():
            raise FileNotFoundError(f"Class mapping not found at {mapping_path}. Run data processing first!")
            
        with open(mapping_path, "r") as f:
            self.class_mapping = json.load(f)
            
        # Create inverse mapping (index -> string name)
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Load the dynamic multi-class model
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load the existing MultiClassDroneCNN checkpoint."""
        model = MultiClassDroneCNN(num_classes=self.num_classes).to(self.device)

        # Pass None for the optimizer since we are only doing inference
        model, _, epoch, loss = load_checkpoint(model, None, model_path)
        model.eval()

        print(f"Loaded MultiClassDroneCNN checkpoint (epoch={epoch}, loss={loss:.6f})")
        return model

    @staticmethod
    def prepare_windows(windows: np.ndarray) -> torch.Tensor:
        """
        Convert complex IQ windows into shape [B, 2, 4096].
        Channel 0: Real, Channel 1: Imaginary.
        """
        batch = np.empty((len(windows), 2, windows.shape[1]), dtype=np.float32)

        for index, window in enumerate(windows):
            window = normalize(window)
            batch[index, 0] = window.real
            batch[index, 1] = window.imag

        return torch.from_numpy(batch)

    def predict(self, iq: np.ndarray) -> dict:
        """
        Run inference on a complex IQ recording/chunk.
        """
        if not np.iscomplexobj(iq):
            raise ValueError("IQ input must be complex.")

        windows = create_windows(iq)
        if len(windows) == 0:
            raise ValueError("Not enough IQ samples for one inference window.")

        all_preds = []

        # Batch inference loop
        for start in range(0, len(windows), self.batch_size):
            batch_windows = windows[start:start + self.batch_size]
            
            batch = self.prepare_windows(batch_windows).to(self.device)

            with torch.no_grad():
                logits = self.model(batch)
                # Multi-class uses argmax instead of sigmoid thresholding
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        # Most common class predicted across all windows in this chunk
        counter = Counter(all_preds)
        most_common_idx = counter.most_common(1)[0][0]
        prediction_str = self.idx_to_class[most_common_idx]
        
        # Calculate a pseudo "confidence" based on how many windows agreed
        agreement_ratio = counter[most_common_idx] / len(all_preds)

        return {
            "windows": len(all_preds),
            "predicted_index": int(most_common_idx),
            "prediction_str": prediction_str,
            "agreement_ratio": agreement_ratio,
        }