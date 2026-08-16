from data_processing.split import split

def test_split_exclusivity():
    """Test that a file only ever appears in one split."""
    files = ["file1.bin", "file2.bin", "file3.bin", "file4.bin", "file5.bin", "file6.bin", "file7.bin", "file8.bin", "file9.bin", "file10.bin"]
    
    train, val, test = split(files, 0.8, 0.1)
    
    # 8 files in train, 1 in val, 1 in test
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1
    
    # Check for exclusivity
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

def test_split_single_file_goes_to_train():
    """Test that if a class only has 1 file, it is strictly placed in the training set."""
    files = ["lonely_drone.bin"]
    
    train, val, test = split(files, 0.8, 0.1)
    
    assert len(train) == 1
    assert len(val) == 0
    assert len(test) == 0
    
    assert train[0] == "lonely_drone.bin"
