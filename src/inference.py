import torch
from torchvision import transforms
from PIL import Image
from src.utils import load_trained_model

# Load the trained model
MODEL_PATH = "results/trained_model.pth"
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("results/trained_model.pth", 2, device)


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

# Run inference if executed directly
if __name__ == "__main__":
    image_path = "datasets/sample_image.jpg"  # Change this path to your test image
    predicted_label = predict(image_path)
    print(f"Predicted Class: {predicted_label}")
