import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, train_ratio=0.8):
    """
    Load images from the dataset directory and apply transformations.

    Args:
        data_dir (str): Path to dataset folder.
        batch_size (int): Batch size for training and validation.
        train_ratio (float): Percentage of dataset to use for training.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images for CNN input
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to range [-1,1]
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
