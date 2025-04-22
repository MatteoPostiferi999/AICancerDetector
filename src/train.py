import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from src.metrics import compute_metrics


def train_model(model, dataloader, criterion, optimizer, device, epochs=10, use_wandb=False):
    # 🕒 Crea una cartella unica per ogni run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    RUN_DIR = os.path.join("results", f"run_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"📁 Risultati salvati in: {RUN_DIR}")

    # 🚀 Inizializza wandb se attivo
    if use_wandb:
        import wandb
        wandb.init(project="IDC-binary-classification", name=f"run_{timestamp}")

    history = {k: [] for k in ["loss", "accuracy", "precision", "recall", "f1"]}
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

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

        print(f"📈 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

        # 💾 Salva il best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))

        # 📤 Log su Weights & Biases
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss, **metrics})

    # 💾 Salva modello finale e storico metriche
    torch.save(model.state_dict(), os.path.join(RUN_DIR, "histology_model.pth"))

    with open(os.path.join(RUN_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # 📝 Info run
    with open(os.path.join(RUN_DIR, "run_info.txt"), "w") as f:
        f.write(f"Epoche: {epochs}\nBest F1: {best_f1:.4f}\nTimestamp: {timestamp}\n")

    # 📈 Grafico metriche
    plt.figure(figsize=(10, 6))
    for k in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("📈 Metriche durante il training")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RUN_DIR, "training_metrics_curve.png"))
    plt.close()

    # 🔚 Chiude la sessione wandb
    if use_wandb:
        wandb.finish()

    return history, RUN_DIR
