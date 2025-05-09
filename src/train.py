import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from src.metrics import compute_metrics
import os

# Percorso persistente in Drive dove salvare i checkpoint
CHECKPOINT_DIR = "/content/drive/MyDrive/histo_project/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



def save_checkpoint(model, optimizer, epoch, history, best_state_dict, best_f1):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_f1": best_f1,
        "best_state_dict": best_state_dict
    }
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint salvato a epoca {epoch + 1} → {path}")


def load_latest_checkpoint(model, optimizer, device):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_")]
    if not checkpoints:
        return 0, None, 0.0, None

    latest_ckpt = sorted(checkpoints)[-1]
    ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1
    history = checkpoint["history"]
    best_f1 = checkpoint["best_f1"]
    best_state_dict = checkpoint["best_state_dict"]

    print(f"🔁 Ripreso da checkpoint: {ckpt_path} (epoca {epoch})")
    return epoch, history, best_f1, best_state_dict



def train_model(model, dataloader, criterion, optimizer, device, epochs=10, resume=False):
    from datetime import datetime
    import os
    import json
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from src.metrics import compute_metrics

    # Percorso dove salvare i risultati finali (grafici, modelli)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Percorso per i checkpoint intermedi su Drive
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    start_epoch = 0
    best_f1 = 0.0
    best_state_dict = model.state_dict()
    history = {k: [] for k in ["loss", "accuracy", "precision", "recall", "f1"]}
    patience = 5
    epochs_no_improve = 0

    if resume:
        # 🔥 Carica da Google Drive, non da run_dir
        start_epoch, history, best_f1, best_state_dict = load_latest_checkpoint(model, optimizer, device)
        if history is None:
            history = {k: [] for k in ["loss", "accuracy", "precision", "recall", "f1"]}
        print(f"🔁 Training ripreso da epoca {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        metrics = compute_metrics(y_true, y_pred)
        history["loss"].append(avg_loss)
        for k in metrics:
            history[k].append(metrics[k])

        print(f"📈 Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

        # 🔥 aggiorna best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⛔ Early stopping attivato dopo {patience} epoche senza miglioramenti.")
                break


        # 🔥 salva su Drive ogni 5 epoche
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, history, best_state_dict, best_f1)

    # ✅ Salvataggio finale su /content (grafici, modelli)
    torch.save(model.state_dict(), os.path.join(run_dir, "histology_model.pth"))
    torch.save(best_state_dict, os.path.join(run_dir, "best_model.pth"))

    with open(os.path.join(run_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    with open(os.path.join(run_dir, "run_info.txt"), "w") as f:
        f.write(f"Epoche: {epochs}\nBest F1: {best_f1:.4f}\nResume: {resume}\n")


    print(f"✅ Training completato. Risultati salvati in {run_dir}")
    return history, run_dir
