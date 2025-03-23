import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Definisci le trasformazioni da applicare alle immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize standard per CNN
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Converte in tensor
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalizzazione per 3 canali RGB
])


# Funzione per caricare dataset + dataloader

def get_dataloader(data_dir=None, batch_size=32, shuffle=True):
    if data_dir is None:
        # Usa il path corretto a seconda dell'ambiente
        if "COLAB_GPU" in os.environ:
            data_dir = "/content/dataset"
        else:
            data_dir = "datasets/dataset_prepared/train"

        batch_size=32,
        shuffle=True

    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, dataloader
