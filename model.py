import torch.nn as nn
import torch.nn.functional as F
import torch


def get_model(num_classes: int) -> nn.Module:
    """A simple, lightweight CNN model for faster experiments."""
    class SimpleCNN(nn.Module):
        def __init__(self, in_channels: int = 1, num_classes: int = 8) -> None:
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    return SimpleCNN(num_classes=num_classes)
