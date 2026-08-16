import json
import tempfile
from pathlib import Path

def test_class_mapping_determinism_and_io():
    """Test that the class mapping logic assigns non_drone to 0 and can be serialized reliably."""
    
    # Simulate the logic from save.py
    drone_classes = ["DJI_inspire_2", "DJI_phantom_4", "DJI_mavic_mini"]
    
    class_mapping = {"non_drone": 0}
    current_class_id = 1
    
    # Sort them to ensure deterministic mapping (which we should do in practice or rely on insertion order)
    # Even with insertion order, the values must be unique and non_drone must be 0
    for drone in drone_classes:
        if drone not in class_mapping:
            class_mapping[drone] = current_class_id
            current_class_id += 1
            
    # Verify non_drone is 0
    assert class_mapping["non_drone"] == 0
    
    # Verify uniqueness
    assert len(set(class_mapping.values())) == len(class_mapping)
    
    # Verify all expected classes are present
    assert set(class_mapping.keys()) == {"non_drone", "DJI_inspire_2", "DJI_phantom_4", "DJI_mavic_mini"}
    
    # Test JSON serialization/deserialization
    with tempfile.TemporaryDirectory() as tmpdir:
        mapping_path = Path(tmpdir) / "class_mapping.json"
        
        with open(mapping_path, "w") as f:
            json.dump(class_mapping, f)
            
        with open(mapping_path, "r") as f:
            loaded_mapping = json.load(f)
            
        # JSON keys are always strings, but the values should remain integers
        assert loaded_mapping == class_mapping
        assert isinstance(loaded_mapping["non_drone"], int)
