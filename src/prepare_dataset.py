import os
import shutil
import random
from tqdm import tqdm

def prepare_dataset(
    source_root,
    target_root,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Prepara un dataset per ImageFolder, copiando immagini da una struttura:
    source_root/[id]/[0|1]/img.png
    a una struttura:
    target_root/[train|val|test]/[0|1]/img.png

    Args:
        source_root (str): path alla cartella originale (es. /content/dataset/IDC_regular_ps50_idx5)
        target_root (str): path dove salvare il dataset preparato
        train_ratio (float): proporzione del training set
        val_ratio (float): proporzione del validation set
        test_ratio (float): proporzione del test set
        seed (int): seme random per mescolamento
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Le proporzioni devono sommare a 1"

    # Crea cartelle target
    for split in ['train', 'val', 'test']:
        for class_label in ['0', '1']:
            os.makedirs(os.path.join(target_root, split, class_label), exist_ok=True)

    # Raccoglie tutte le immagini
    all_images = []

    for folder_name in os.listdir(source_root):
        folder_path = os.path.join(source_root, folder_name)
        if os.path.isdir(folder_path):
            for class_label in ['0', '1']:
                class_path = os.path.join(folder_path, class_label)
                if os.path.exists(class_path):
                    for file_name in os.listdir(class_path):
                        src_path = os.path.join(class_path, file_name)
                        all_images.append((src_path, class_label))

    print(f"🔍 Trovate {len(all_images)} immagini totali")

    # Mescola immagini
    random.seed(seed)
    random.shuffle(all_images)

    # Split
    total = len(all_images)
    train_cutoff = int(total * train_ratio)
    val_cutoff = int(total * (train_ratio + val_ratio))

    train_set = all_images[:train_cutoff]
    val_set = all_images[train_cutoff:val_cutoff]
    test_set = all_images[val_cutoff:]

    # Funzione interna per copiare le immagini
    def copy_images(image_list, split_name):
        for src_path, label in tqdm(image_list, desc=f'Copying {split_name}'):
            filename = os.path.basename(src_path)
            dest_path = os.path.join(target_root, split_name, label, filename)
            shutil.copy2(src_path, dest_path)

    copy_images(train_set, 'train')
    copy_images(val_set, 'val')
    copy_images(test_set, 'test')

    print("✅ Dataset pronto!")

