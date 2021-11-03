# import required modules
import os, torchvision
from typing import Tuple
from torch.utils.data import DataLoader

def loadMNIST() -> Tuple[DataLoader]:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = int(128)

    trainset = torchvision.datasets.MNIST(root=os.path.normpath('~/Documents/Data/MNIST'), train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(root=os.path.normpath('~/Documents/Data/MNIST'), train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
