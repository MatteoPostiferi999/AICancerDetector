# src/train.py

import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.metrics import compute_metrics

def train_model(model, dataloader, criterion, optimizer, device, epochs=10, save_path="results/", use_wandb=False):
    os.makedirs(save_path, exist_ok=True)
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

        # 💾 Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "loss": avg_loss, **metrics})

    # 💾 Save final model
    torch.save(model.state_dict(), os.path.join(save_path, "histology_model.pth"))
    with open(os.path.join(save_path, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # 📈 Plot
    plt.figure(figsize=(10, 6))
    for k in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("📈 Metriche durante il training")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, "training_metrics_curve.png"))
    plt.close()

    return history
