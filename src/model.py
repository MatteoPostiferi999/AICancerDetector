import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes):
    """
    Load a pre-trained CNN and modify the final layer for classification.

    Args:
        num_classes (int): Number of output classes (tumor types/stages).

    Returns:
        model (torch.nn.Module): Modified CNN model.
    """
    model = models.resnet18(pretrained=True)  # Load ResNet18
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust output layer

    return model
