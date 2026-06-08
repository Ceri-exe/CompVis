import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST digit classification.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flattening the image tensor if not already done
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        pred = self.out(x)
        return pred