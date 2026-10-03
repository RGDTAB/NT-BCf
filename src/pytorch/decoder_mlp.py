import torch
import torch.nn as nn

# Pytorch MLP with a single hidden layer, and a ReLU activation after that hidden layer
class DecoderMLP(nn.Module):
    def __init__(self, input_channels: int, hidden_width: int, output_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(input_channels, hidden_width)
        self.lin2 = nn.Linear(hidden_width, output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.lin2(self.relu(self.lin1(x)))

