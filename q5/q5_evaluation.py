"""
Question 5: Deep learning evaluation metrics and plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Outputs are organized in outputs/q5/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q5')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(model, test_loader, device=None, class_names=None):
    """
    Evaluate model on test data.

    Returns:
        metrics: dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    total = 0
    correct = 0
    top5_correct = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            total += targets.size(0)
            correct += (preds == targets).sum().item()

            top5 = torch.topk(outputs, k=5, dim=1).indices
            top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    accuracy = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    top5_acc = top5_correct / total if total > 0 else 0.0
    top5_error = 1.0 - top5_acc

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top5_error": top5_error,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot and save confusion matrix.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'q5_confusion_matrix.jpg')

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(train_loss, val_loss, train_acc, val_acc, save_path=None):
    """
    Plot training/validation loss and accuracy curves.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'q5_learning_curves.jpg')

    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, label='Train Loss')
    axes[0].plot(epochs, val_loss, label='Val Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label='Train Acc')
    axes[1].plot(epochs, val_acc, label='Val Acc')
    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
