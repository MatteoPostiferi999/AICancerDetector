import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, train_ratio=0.8):
    """
    Loads images from the dataset directory, applies transformations,
    and splits the dataset into training and validation sets.

    Args:
        data_dir (str): Path to the dataset folder containing images.
        batch_size (int): Number of images per batch.
        train_ratio (float): Proportion of the dataset to use for training.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    """

    # Define the transformations to apply to each image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels for CNN input
        transforms.ToTensor(),          # Convert images to PyTorch tensors (scale [0,255] to [0,1])
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to the range [-1,1]
    ])

    # Load the dataset using ImageFolder (assumes subdirectories as class labels)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Compute the number of images for training and validation
    train_size = int(train_ratio * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # The rest (20%) for validation

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders to load data in batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle training data for randomness
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Do not shuffle validation data

    return train_loader, val_loader  # Return DataLoaders for training and validation