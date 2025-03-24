import os
import shutil
import random
from tqdm import tqdm


def prepare_dataset(
    origin_root="/content/dataset/IDC_regular_ps50_idx5",
    dest_root="/content/dataset_prepared",

    train_pct=0.7,
    val_pct=0.15,
    test_pct=0.15
):

    """
    Prepara un dataset istologico dividendo le immagini in
    train/val/test e classificandole in IDC e non-IDC in base al nome file.

    Args:
        origin_root (str): percorso delle immagini originali grezze
        dest_root (str): cartella di destinazione strutturata
        train_pct (float): percentuale per training set
        val_pct (float): percentuale per validation set
        test_pct (float): percentuale per test set
    """
    print("Provaaaaa1")
    # 🧹 Elimina cartella esistente
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)

    # 📁 Crea la struttura di destinazione
    for split in ["train", "val", "test"]:
        for label in ["IDC", "non-IDC"]:
            os.makedirs(os.path.join(dest_root, split, label))

    # 🖼️ Leggi tutte le immagini e classifica in base al nome
    all_images = []
    print("Provaaa2")
    for subdir in os.listdir(origin_root):
        print("Provaaa3")

        sub_path = os.path.join(origin_root, subdir)
        print("Provaaa4")

        if os.path.isdir(sub_path):
            for img_name in os.listdir(sub_path):
                full_path = os.path.join(sub_path, img_name)

                if "class0" in img_name:
                    label = "non-IDC"
                elif "class1" in img_name:
                    label = "IDC"
                else:
                    continue

                all_images.append((full_path, label))

    print(f"🔍 Trovate {len(all_images)} immagini totali")

    # 🔀 Mescola in modo casuale
    random.shuffle(all_images)

    # ✂️ Calcola gli split
    train_end = int(len(all_images) * train_pct)
    val_end = train_end + int(len(all_images) * val_pct)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    # 📥 Copia le immagini
    for split, samples in splits.items():
        print(f"📂 Copiando {split} ({len(samples)} immagini)")
        for src_path, label in tqdm(samples):
            dest_path = os.path.join(dest_root, split, label, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)

    print("✅ Dataset preparato con successo!")

