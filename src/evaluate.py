import torch
from src.dataset_loader import get_dataloader
from src.utils import load_trained_model

# Load dataset
_, val_loader = get_dataloader(batch_size=32)  # Assuming get_dataloader() returns train & val loaders

# Load model
MODEL_PATH = "results/trained_model.pth"
NUM_CLASSES = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("results/trained_model.pth", 3, device)

# Evaluation function
def evaluate_model(model, dataloader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Model Accuracy: {accuracy:.2f}%")
    return accuracy

# Run evaluation
if __name__ == "__main__":
    evaluate_model(model, val_loader)
