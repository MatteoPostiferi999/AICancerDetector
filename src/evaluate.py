import os
import torch
import json
from src.dataset_loader import get_dataloader
from src.utils import load_trained_model
from src.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curve

# 📦 Configurazione
MODEL_PATH = "results/trained_model.pth"
NUM_CLASSES = 2
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔁 Funzione di evaluation completa
def evaluate_model(model, dataloader, save_dir=RESULTS_DIR):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Per multiclass
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())  # Tutte le classi

    # 🔍 Metriche
    metrics = compute_metrics(y_true, y_pred, average="macro")  # per multiclass
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 💾 Salva metriche in JSON
    with open(os.path.join(save_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 📊 Grafici
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(save_dir, "val_confusion_matrix.png"), labels=None)

    if NUM_CLASSES == 2:
        plot_roc_curve(y_true, [p[1] for p in y_scores], save_path=os.path.join(save_dir, "val_roc_curve.png"))
    else:
        print("ℹ️ ROC/AUC plotting skipped for multiclass. Implementazione futura.")

    return metrics


# 🧪 Esegui solo se usato come script
if __name__ == "__main__":
    # Carica dati e modello
    _, val_loader = get_dataloader(batch_size=32)
    model = load_trained_model(MODEL_PATH, NUM_CLASSES, device)

    evaluate_model(model, val_loader)
