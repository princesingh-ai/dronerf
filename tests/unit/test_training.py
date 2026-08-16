import torch
import torch.nn as nn
from training.train import train_one_epoch
from training.validate import validate
from training.config import DEVICE

class MockModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(10, num_classes)
        
    def forward(self, x):
        # x is [batch, 2, samples], we flatten and project it
        # For mock purposes, just return a random logit tensor that requires grad
        batch_size = x.size(0)
        return self.linear(torch.randn(batch_size, 10, device=x.device))


def test_training_and_validation_logic():
    """Test the train_one_epoch and validate functions using synthetic dataloaders."""
    
    num_classes = 4
    batch_size = 2
    model = MockModel(num_classes=num_classes).to(DEVICE)
    
    # Save initial weights to check for updates
    initial_weight = model.linear.weight.clone()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Mock dataloader: yields (windows, labels)
    # windows: [batch, 2, 4096], labels: [batch] (torch.long)
    mock_dataloader = [
        (torch.randn(batch_size, 2, 4096), torch.tensor([0, 2], dtype=torch.long)),
        (torch.randn(batch_size, 2, 4096), torch.tensor([1, 3], dtype=torch.long))
    ]
    
    # Test Training
    loss = train_one_epoch(model, mock_dataloader, criterion, optimizer)
    
    assert isinstance(loss, float)
    assert loss > 0
    
    # Ensure weights were updated!
    assert not torch.equal(model.linear.weight, initial_weight)
    
    # Test Validation
    val_loss = validate(model, mock_dataloader, criterion)
    
    assert isinstance(val_loss, float)
    assert val_loss > 0
    
    # Ensure predictions output shape [batch, num_classes] for CrossEntropy
    # This proves the loss function works natively without binary unsqueeze hacks
    outputs = model(mock_dataloader[0][0].to(DEVICE))
    assert outputs.shape == (batch_size, num_classes)
    
    # Ensure we can argmax it (like in evaluate.py)
    preds = torch.argmax(outputs, dim=1)
    assert preds.shape == (batch_size,)
    assert preds.dtype == torch.long
