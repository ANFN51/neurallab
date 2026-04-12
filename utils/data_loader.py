# utils/data_loader.py
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import ssl

# Fix for macOS SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def get_data_loaders(batch_size=64):
    """Load MNIST dataset with SSL fix and better messages"""

    os.makedirs("./data/MNIST/raw", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading MNIST dataset... (this may download ~11MB the first time)")

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("MNIST dataset loaded successfully!")
    return train_loader, test_loader