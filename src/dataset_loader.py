import os
import tarfile
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define dataset path
dataset_path = "cancer_dataset"
compressed_file = "cancer_dataset.tar.gz"

# Extract dataset if not already extracted
if not os.path.exists(dataset_path):
    with tarfile.open(compressed_file, "r:gz") as tar:
        tar.extractall(path=".")
    print("Dataset extracted!")


# Define PyTorch Dataset class
class HistologyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_map = {name: i for i, name in enumerate(os.listdir(root_dir))}

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Function to get DataLoader
def get_dataloader(batch_size=32, shuffle=True):
    dataset = HistologyDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Dataset size: {len(dataset)}")

    # Save dataset and dataloader
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    with open("dataloader.pkl", "wb") as f:
        pickle.dump(dataloader, f)

    return dataset, dataloader
