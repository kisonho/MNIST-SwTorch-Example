# import required modules
import torch
from torch.nn import functional as F

class CNN(torch.nn.Module):
    # layers
    conv1: torch.nn.Conv2d
    pool1: torch.nn.MaxPool2d
    conv2: torch.nn.Conv2d
    pool2: torch.nn.MaxPool2d
    conv3: torch.nn.Conv2d
    pool3: torch.nn.MaxPool2d
    fc1: torch.nn.Linear
    classification: torch.nn.Linear
    
    # constructor
    def __init__():
        super().__init__()
        
        # initialize layers
        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(128, 256, 3)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(256, 512)
        self.classification = torch.nn.Linear(512, 10)

    # forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.fc1(x)
        x = F.relu(x)
        y = self.classification(x)
        return y
