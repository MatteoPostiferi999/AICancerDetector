import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from src.model import build_model

# Load the trained model
MODEL_PATH = "results/trained_model.pth"
NUM_CLASSES = 3  # Adjust based on dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict(image_path):
    """
    Perform inference on a single image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Predicted class label.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Example usage
image_path = "datasets/sample_image.jpg"
predicted_label = predict(image_path)
print(f"Predicted Class: {predicted_label}")
