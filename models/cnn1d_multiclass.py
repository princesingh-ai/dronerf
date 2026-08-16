import torch
import torch.nn as nn

# We are keeping the exact same 1D CNN feature extractor backbone as the binary model.
# This ensures that our temporal receptive field and filter sizes (which we know work well
# for RF signals) remain untouched. The only difference is the dynamically sized classifier head.
class MultiClassDroneCNN(nn.Module):
    """1D CNN for multi-class drone RF signal identification."""

    def __init__(self, num_classes: int):
        super().__init__()

        # The feature extraction backbone remains identical to preserve research continuity
        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )

        # The classifier now outputs `num_classes` logits instead of 1.
        # We will use CrossEntropyLoss during training which handles the Softmax implicitly.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
