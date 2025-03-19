import torch
import torch.optim as optim
import torch.nn as nn
from src.dataset_loader import get_data_loaders
from src.model import build_model

# Define parameters
DATA_DIR = "datasets/TCIA"  # Change this to your dataset path
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Adjust based on tumor types

# Load data
train_loader, val_loader = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training complete!")
torch.save(model.state_dict(), "results/trained_model.pth")
