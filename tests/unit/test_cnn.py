import torch
from models.cnn1d_multiclass import MultiClassDroneCNN

def test_cnn_architecture():
    """Test that the CNN accepts [batch, 2, samples] and outputs [batch, num_classes]."""
    batch_size = 4
    num_samples = 4096
    num_classes = 8
    
    # 2 channels (Real, Imaginary)
    mock_input = torch.randn(batch_size, 2, num_samples)
    
    model = MultiClassDroneCNN(num_classes=num_classes)
    
    # Forward pass
    output = model(mock_input)
    
    # Check output shape
    assert output.shape == (batch_size, num_classes)
    
    # Check gradients flow
    loss = output.sum()
    loss.backward()
    
    # Check that parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None