import os
import shutil
import random
from tqdm import tqdm

# 📂 Path originale (da Kaggle)
origin_root = "/content/dataset/IDC_regular_ps50_idx5"

# 📂 Path destinazione per PyTorch (ImageFolder-ready)
dest_root = "/content/dataset_prepared"

# ✨ Percentuali split
train_pct, val_pct, test_pct = 0.7, 0.15, 0.15

# 🧹 Elimina cartella se già esiste
if os.path.exists(dest_root):
    shutil.rmtree(dest_root)

# 📁 Crea le sottocartelle
for split in ["train", "val", "test"]:
    for label in ["IDC", "non-IDC"]:
        os.makedirs(os.path.join(dest_root, split, label))

# 🖼️ Raccoglie immagini con label
all_images = []

for subfolder in os.listdir(origin_root):
    full_subfolder = os.path.join(origin_root, subfolder)
    if not os.path.isdir(full_subfolder):
        continue

    for img_name in os.listdir(full_subfolder):
        full_path = os.path.join(full_subfolder, img_name)
        if "class0" in img_name:
            label = "non-IDC"
        elif "class1" in img_name:
            label = "IDC"
        else:
            continue
        all_images.append((full_path, label))

print(f"📦 Trovate {len(all_images)} immagini totali")

# 🔀 Mescola immagini
random.shuffle(all_images)

# ✂️ Divide dataset
train_end = int(len(all_images) * train_pct)
val_end = train_end + int(len(all_images) * val_pct)

splits = {
    "train": all_images[:train_end],
    "val": all_images[train_end:val_end],
    "test": all_images[val_end:]
}

# 📥 Copia immagini in dataset_prepared/
for split, samples in splits.items():
    print(f"✏️ Copiando {split} ({len(samples)} immagini)")
    for src_path, label in tqdm(samples):
        dest_path = os.path.join(dest_root, split, label, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)

print("✅ Dataset pronto in /content/dataset_prepared/")
