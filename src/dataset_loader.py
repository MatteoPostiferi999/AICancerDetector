import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataloader function using ImageFolder
def get_dataloader(data_dir=None, batch_size=32, shuffle=True):
    if data_dir is None:
        data_dir = (
            "/content/dataset_prepared/train"
            if "COLAB_GPU" in os.environ
            else "datasets/dataset_prepared/train"
        )

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Path not found: {data_dir}")

    dataset = ImageFolder(
        root=data_dir,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=4, pin_memory=True)
    return dataset, dataloader
