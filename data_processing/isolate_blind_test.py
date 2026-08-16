import shutil
import random
from pathlib import Path

# Move 10% of the raw .bin files to a separate folder before any preprocessing.
# This ensures we don't accidentally leak the blind test data into the training pipeline.
def isolate_blind_test(drone_dir: str, non_drone_dir: str, blind_test_dir: str, ratio: float = 0.1, seed: int = 42):
    # Fixed seed for reproducibility.
    random.seed(seed)
    
    drone_path = Path(drone_dir)
    non_drone_path = Path(non_drone_dir)
    blind_test_path = Path(blind_test_dir)
    
    blind_test_drone = blind_test_path / "drone_rf"
    blind_test_non_drone = blind_test_path / "random_rf"
    
    blind_test_drone.mkdir(parents=True, exist_ok=True)
    blind_test_non_drone.mkdir(parents=True, exist_ok=True)

    # Grab all drone files and shuffle them
    drone_files = [f for f in drone_path.glob("*.bin") if f.is_file()]
    random.shuffle(drone_files)
    num_drone_blind = int(len(drone_files) * ratio)
    
    # Move the selected files out of the main dataset
    for f in drone_files[:num_drone_blind]:
        print(f"Moving {f.name} to blind test")
        shutil.move(str(f), str(blind_test_drone / f.name))

    # Do the same for the background noise/non-drone data
    non_drone_files = [f for f in non_drone_path.rglob("*") if f.is_file() and not f.name.startswith(".DS_Store") and f.suffix != ".txt"]
    random.shuffle(non_drone_files)
    num_non_drone_blind = max(1, int(len(non_drone_files) * ratio)) if non_drone_files else 0
    
    for f in non_drone_files[:num_non_drone_blind]:
        print(f"Moving {f.name} to blind test")
        
        # Maintain the original folder structure (like LTE or Wi-Fi) inside the blind test folder
        rel_path = f.relative_to(non_drone_path)
        dest_path = blind_test_non_drone / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest_path))

if __name__ == "__main__":
    isolate_blind_test(
        drone_dir="drone_rf",
        non_drone_dir="random_rf",
        blind_test_dir="blind_test_raw",
        ratio=0.1
    )
