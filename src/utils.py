import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.model import build_model


def show_batch(dataloader, classes):
    """Displays a batch of images with labels."""
    images, labels = next(iter(dataloader))
    images = torchvision.utils.make_grid(images)
    images = images.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(images)
    plt.title([classes[label] for label in labels])
    plt.axis("off")
    plt.show()


def count_parameters(model):
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_trained_model(model_path, num_classes, device):
    """Loads a trained PyTorch model from a file."""
    model = build_model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


