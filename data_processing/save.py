import json
import re
from pathlib import Path
import numpy as np
import shutil

from data_processing.loaders import load_signal
from data_processing.preprocessing import create_windows
from data_processing.resample import resample_signal
from utils.config import config

def parse_drone_model(filename: str) -> str:
    match = re.match(r"(.*)_(2G|5G).*", filename)
    if match:
        return match.group(1)
    return Path(filename).stem

def create_dataset():
    drone_dir = Path(config["dataset"]["raw_drone_path"])
    non_drone_dir = Path(config["dataset"]["raw_random_path"])
    out_dir = Path(config["dataset"]["processed_path"])
    
    # Wipe the existing output directories to ensure a clean slate
    for d in [out_dir, Path(config["evaluation"]["blind_test_20_path"]), Path(config["evaluation"]["blind_test_60_path"])]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    class_files = {"non_drone": []}
    class_mapping = {"non_drone": 0}
    current_class_id = 1

    for file in non_drone_dir.rglob("*"):
        if file.is_file() and file.suffix in [".bin", ".dat", ".data"]:
            class_files["non_drone"].append(file)

    for file in drone_dir.glob("*.bin"):
        if file.is_file():
            model_name = parse_drone_model(file.name)
            if model_name not in class_files:
                class_files[model_name] = []
                class_mapping[model_name] = current_class_id
                current_class_id += 1
            class_files[model_name].append(file)

    with open(config["dataset"]["mapping_path"], "w") as f:
        json.dump(class_mapping, f, indent=4)
        
    print(f"Discovered classes and generated mapping: {class_mapping}")

    original_rate = config["dataset"]["original_rate"]
    target_rate = config["dataset"]["target_rate"]
    
    # Pass 1: Process all files and concatenate windows in memory per class
    class_windows_20 = {}
    class_windows_60 = {}
    
    for class_name, files in class_files.items():
        print(f"\nLoading and processing files for class: {class_name}")
        windows_20_list = []
        windows_60_list = []
        
        for file in files:
            print(f"  -> Reading {file.name}")
            iq = load_signal(str(file))
            
            # 60 Msps windows (for blind test 60 variant)
            w_60 = create_windows(iq)
            windows_60_list.append(w_60)
            
            # 20 Msps windows (for everything else)
            iq_20 = resample_signal(iq, original_rate=original_rate, target_rate=target_rate)
            w_20 = create_windows(iq_20)
            windows_20_list.append(w_20)
            
        # Concatenate all windows for this class
        if len(windows_20_list) > 0:
            class_windows_20[class_name] = np.concatenate(windows_20_list, axis=0)
            class_windows_60[class_name] = np.concatenate(windows_60_list, axis=0)
            print(f"  Total 20Msps windows for {class_name}: {len(class_windows_20[class_name])}")
        else:
            print(f"  Warning: No data for {class_name}")

    # Find the minimum length to perfectly balance the dataset
    min_windows = min(len(w) for w in class_windows_20.values())
    print(f"\nBalancing all classes to exactly {min_windows} windows (20Msps).")

    train_r = config["dataset"]["train_ratio"]
    val_r = config["dataset"]["val_ratio"]
    test_r = config["dataset"]["test_ratio"]
    blind_r = config["dataset"]["blind_ratio"]
    
    t_end = int(min_windows * train_r)
    v_end = t_end + int(min_windows * val_r)
    te_end = v_end + int(min_windows * test_r)

    # Note: 60Msps arrays have 3x the windows (since rate is 3x), so we scale the indices
    scale = int(original_rate / target_rate)
    
    for class_name in class_windows_20.keys():
        print(f"Slicing and saving splits for {class_name}...")
        
        # 1. Truncate to exact equal length
        arr_20 = class_windows_20[class_name][:min_windows]
        arr_60 = class_windows_60[class_name][:min_windows * scale]
        
        # 2. Slice the 20Msps arrays
        train_20 = arr_20[:t_end]
        val_20 = arr_20[t_end:v_end]
        test_20 = arr_20[v_end:te_end]
        blind_20 = arr_20[te_end:]
        
        # 3. Slice the 60Msps array (only need the blind portion)
        te_end_60 = te_end * scale
        blind_60 = arr_60[te_end_60:]

        # 4. Save to disk
        splits = [
            (out_dir / "train", train_20),
            (out_dir / "validation", val_20),
            (out_dir / "test", test_20),
            (Path(config["evaluation"]["blind_test_20_path"]), blind_20),
            (Path(config["evaluation"]["blind_test_60_path"]), blind_60)
        ]
        
        for base_dir, data in splits:
            class_dir = base_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            np.save(class_dir / f"{class_name}_data.npy", data)
            
            # Save metadata
            meta_path = class_dir / f"{class_name}_data_meta.json"
            meta = {
                "drone_model": class_name,
                "sampling_rate": target_rate if "60Msps" not in str(base_dir) else original_rate,
                "num_windows": len(data)
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)
                
    print("\nSuccessfully generated fully balanced, 70/10/10/10 sequential splits!")

if __name__ == "__main__":
    create_dataset()
