import os
import shutil
import random
from tqdm import tqdm

def prepare_dataset(
    origin_root,
    dest_root,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    # Class label mapping
    label_map = {
        "0": "non-IDC",
        "1": "IDC"
    }

    # Crea cartelle di destinazione
    for split in ['train', 'val', 'test']:
        for label in label_map.values():
            os.makedirs(os.path.join(dest_root, split, label), exist_ok=True)

    # Raccoglie tutte le immagini
    all_images = []

    for folder in os.listdir(origin_root):
        folder_path = os.path.join(origin_root, folder)
        if os.path.isdir(folder_path):
            for class_id, class_name in label_map.items():
                class_path = os.path.join(folder_path, class_id)
                if os.path.exists(class_path):
                    for filename in os.listdir(class_path):
                        img_path = os.path.join(class_path, filename)
                        all_images.append((img_path, class_name))

    print(f"🔍 Trovate {len(all_images)} immagini totali")

    # Mescola immagini
    random.seed(seed)
    random.shuffle(all_images)

    # Split
    total = len(all_images)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))

    train_set = all_images[:train_cut]
    val_set = all_images[train_cut:val_cut]
    test_set = all_images[val_cut:]

    # Copia le immagini nella struttura giusta
    def copy_images(image_list, split_name):
        for src_path, label in tqdm(image_list, desc=f"📁 Copiando {split_name}"):
            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_root, split_name, label, filename)
            shutil.copy2(src_path, dest_path)

    copy_images(train_set, 'train')
    copy_images(val_set, 'val')
    copy_images(test_set, 'test')

    print("✅ Dataset preparato correttamente!")

