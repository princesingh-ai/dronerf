import json
import re
from pathlib import Path
import numpy as np
from data_processing.loaders import load_signal
from data_processing.preprocessing import create_windows
from data_processing.resample import resample_signal


def extract_class_name(filename: str) -> str:
    """Extract drone model class name from filename."""
    if "non_drone" in filename.lower() or "random" in filename.lower():
        return "non_drone"
    
    # E.g., DJI_inspire_2_2G.bin -> DJI_inspire_2
    match = re.match(r"(.*?)_\dG", filename)
    if match:
        return match.group(1)
        
    return Path(filename).stem


def process_blind_file(
    file_path: Path, 
    output_dir_20: Path, 
    output_dir_60: Path,
):
    class_name = extract_class_name(file_path.name)
    
    # Create class-specific subdirectories
    class_dir_20 = output_dir_20 / class_name
    class_dir_60 = output_dir_60 / class_name
    
    class_dir_20.mkdir(parents=True, exist_ok=True)
    class_dir_60.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the original 60 Msps raw signal
    iq_60 = load_signal(str(file_path))
    
    # ---------------------------------------------------------
    # Process 60 Msps Version
    # ---------------------------------------------------------
    windows_60 = create_windows(iq_60)
    output_path_60 = class_dir_60 / f"{file_path.stem}.npy"
    np.save(output_path_60, windows_60)
    
    meta_60 = {
        "original_filename": file_path.name,
        "drone_model": class_name,
        "sampling_rate": 60e6,
    }
    with open(class_dir_60 / f"{file_path.stem}_meta.json", "w") as f:
        json.dump(meta_60, f, indent=4)
        
    # ---------------------------------------------------------
    # Process 20 Msps Version
    # ---------------------------------------------------------
    iq_20 = resample_signal(iq_60, original_rate=60_000_000, target_rate=20_000_000)
    windows_20 = create_windows(iq_20)
    output_path_20 = class_dir_20 / f"{file_path.stem}.npy"
    np.save(output_path_20, windows_20)
    
    meta_20 = {
        "original_filename": file_path.name,
        "drone_model": class_name,
        "sampling_rate": 20e6,
    }
    with open(class_dir_20 / f"{file_path.stem}_meta.json", "w") as f:
        json.dump(meta_20, f, indent=4)


def process_blind_test(input_dir: str, out_dir_20: str, out_dir_60: str, mapping_path: str):
    input_path = Path(input_dir)
    dir_20 = Path(out_dir_20)
    dir_60 = Path(out_dir_60)
    
    dir_20.mkdir(parents=True, exist_ok=True)
    dir_60.mkdir(parents=True, exist_ok=True)
    
    # We must copy the class_mapping.json so RFDataset can read it
    import shutil
    if Path(mapping_path).exists():
        shutil.copy(mapping_path, dir_20 / "class_mapping.json")
        shutil.copy(mapping_path, dir_60 / "class_mapping.json")
    else:
        print(f"Warning: {mapping_path} not found. Ensure save.py ran first!")
        return

    files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix in [".bin", ".dat", ".data"]]
    
    print(f"Found {len(files)} raw files in blind test set. Processing...")
    
    for file in files:
        print(f"  -> Processing {file.name}")
        process_blind_file(file, dir_20, dir_60)
        
    print("Done! Blind test set successfully branched into 20 Msps and 60 Msps environments.")


if __name__ == "__main__":
    process_blind_test(
        input_dir="blind_test_raw",
        out_dir_20="processed_blind_test_20Msps",
        out_dir_60="processed_blind_test_60Msps",
        mapping_path="processed/class_mapping.json",
    )
