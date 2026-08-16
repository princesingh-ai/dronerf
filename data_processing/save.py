import json
import re
from pathlib import Path
import numpy as np
from data_processing.loaders import load_signal
from data_processing.preprocessing import create_windows
from data_processing.split import split_dataset
from data_processing.resample import resample_signal

# We need to extract the exact drone model from the filename.
# Most drone files in the dataset look like "DJI_inspire_2_2G.bin" or "DJI_phantom_4_pro_plus_5G_1of2.bin".
# I'll use a regex to strip off the band info ("_2G", "_5G") and file parts ("_1of2") to get the pure model name.
def parse_drone_model(filename: str) -> str:
    # Match everything up to the last _2G or _5G
    match = re.match(r"(.*)_(2G|5G).*", filename)
    if match:
        return match.group(1)
    
    # Fallback if the filename doesn't follow the pattern
    return Path(filename).stem

def process_split(
    files: list[Path],
    output_dir: Path,
    class_name: str,
):
    # Ensure the output directory for this specific class exists (e.g. processed/train/DJI_inspire_2/)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        iq = load_signal(str(file))
        
        # Downsample the raw 60 Msps recording to 20 Msps so it matches the live SDR hardware!
        iq = resample_signal(iq, original_rate=60_000_000, target_rate=20_000_000)
        
        windows = create_windows(iq)
        output_path = output_dir / f"{file.stem}.npy"
        
        # Save the sliding windows as an npy file for memory mapping during training
        np.save(output_path, windows)
        
        # We need to preserve the metadata for future research on sampling rates and generalization!
        # I'm saving it alongside the npy file so it doesn't interfere with np.load mmap_mode.
        metadata = {
            "original_filename": file.name,
            "drone_model": class_name,
            "sampling_rate": 20e6, # Downsampled to 20 Msps to match live hardware
        }
        
        meta_path = output_dir / f"{file.stem}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)


def create_dataset(
    drone_path: str,
    non_drone_path: str,
    output_dir: str,
):
    drone_dir = Path(drone_path)
    non_drone_dir = Path(non_drone_path)
    out_dir = Path(output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    class_files = {"non_drone": []}
    class_mapping = {"non_drone": 0}
    current_class_id = 1

    # First, let's collect all the non-drone background noise files
    for file in non_drone_dir.rglob("*"):
        if file.is_file() and file.suffix in [".bin", ".dat", ".data"]:
            class_files["non_drone"].append(file)

    # Now let's dynamically discover all the drone models from the file names
    for file in drone_dir.glob("*.bin"):
        if file.is_file():
            model_name = parse_drone_model(file.name)
            
            if model_name not in class_files:
                class_files[model_name] = []
                class_mapping[model_name] = current_class_id
                current_class_id += 1
                
            class_files[model_name].append(file)

    # I'll save this mapping to disk. This mapping is our single source of truth!
    # The PyTorch Dataset and the Inference script will load this to ensure labels match up.
    mapping_path = out_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=4)
        
    print(f"Generated class mapping: {class_mapping}")

    # Now perform our perfectly stratified multi-class split across all models
    splits = split_dataset(class_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Finally, process and save the windows for each split and class
    for split_name, classes in splits.items():
        for class_name, files in classes.items():
            print(f"Processing {split_name} for class {class_name} ({len(files)} files)...")
            
            class_output_dir = out_dir / split_name / class_name
            process_split(files, class_output_dir, class_name)


if __name__ == "__main__":
    create_dataset(
        drone_path="drone_rf",
        non_drone_path="random_rf",
        output_dir="processed",
    )