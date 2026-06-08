import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for MNIST digit classification.
    Features 2 Convolutional blocks followed by a fully-connected output layer.
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # Block 1: Input (1, 28, 28) -> Output (16, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Block 2: Input (16, 14, 14) -> Output (32, 7, 7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Output Linear Layer: 32 * 7 * 7 flattened features -> 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten/Vectorize
        pred = self.out(x)
        return pred