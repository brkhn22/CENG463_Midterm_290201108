"""
Question 5: Orchestrator for the full PyTorch pipeline.
"""

import os
import torch
import torch.nn as nn

from q5_data import get_dataloaders
from q5_models import DeepMLP, CustomCNN, TransferResNet
from q5_training import train_model
from q5_evaluation import evaluate_model, plot_confusion_matrix, plot_learning_curves
from q5_interpret import plot_gradcam_misclassified, plot_mlp_shap
from q5_adversarial import fgsm_attack_accuracy

# Outputs are organized in outputs/q5/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q5')
os.makedirs(OUTPUT_DIR, exist_ok=True)


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_model(model, train_loader, val_loader, device, name, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loss, val_loss, train_acc, val_acc = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        max_epochs=15,
        patience=5,
    )

    curves_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_learning_curves.jpg")
    plot_learning_curves(train_loss, val_loss, train_acc, val_acc, save_path=curves_path)

    return train_loss, val_loss, train_acc, val_acc


def evaluate_one_model(model, test_loader, device, name):
    metrics = evaluate_model(model, test_loader, device=device, class_names=CIFAR10_CLASSES)

    cm_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_confusion_matrix.jpg")
    plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names=CIFAR10_CLASSES, save_path=cm_path)

    return metrics


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Phase 1: Data
    print("\n[Phase 1] Loading CIFAR-10 dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    # Phase 2 & 3: Models + Training (no Optuna)
    print("\n[Phase 2-3] Initializing models...")
    mlp = DeepMLP(dropout=0.5).to(device)
    cnn = CustomCNN(dropout=0.3).to(device)
    resnet = TransferResNet().to(device)

    print("\nTraining DeepMLP...")
    train_one_model(mlp, train_loader, val_loader, device, name="DeepMLP")

    print("\nTraining CustomCNN...")
    train_one_model(cnn, train_loader, val_loader, device, name="CustomCNN")

    print("\nTraining TransferResNet...")
    train_one_model(resnet, train_loader, val_loader, device, name="TransferResNet")

    # Phase 4: Evaluation
    print("\n[Phase 4] Evaluating models...")
    mlp_metrics = evaluate_one_model(mlp, test_loader, device, name="DeepMLP")
    cnn_metrics = evaluate_one_model(cnn, test_loader, device, name="CustomCNN")
    resnet_metrics = evaluate_one_model(resnet, test_loader, device, name="TransferResNet")

    def print_metrics(label, metrics):
        print(f"{label} -> Acc: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, Top-5 Error: {metrics['top5_error']:.4f}")

    print_metrics("DeepMLP", mlp_metrics)
    print_metrics("CustomCNN", cnn_metrics)
    print_metrics("TransferResNet", resnet_metrics)

    # Phase 5: Interpretability
    print("\n[Phase 5] Interpretability outputs...")
    plot_gradcam_misclassified(cnn, test_loader, class_names=CIFAR10_CLASSES, device=device)
    plot_mlp_shap(mlp, test_loader, class_names=CIFAR10_CLASSES, device=device)

    # Phase 6: Adversarial robustness
    print("\n[Phase 6] FGSM adversarial evaluation...")
    eps = 8 / 255
    mlp_adv_acc = fgsm_attack_accuracy(mlp, test_loader, eps=eps, device=device)
    cnn_adv_acc = fgsm_attack_accuracy(cnn, test_loader, eps=eps, device=device)
    resnet_adv_acc = fgsm_attack_accuracy(resnet, test_loader, eps=eps, device=device)

    print(f"FGSM (eps={eps:.5f}) accuracy -> DeepMLP: {mlp_adv_acc:.4f}")
    print(f"FGSM (eps={eps:.5f}) accuracy -> CustomCNN: {cnn_adv_acc:.4f}")
    print(f"FGSM (eps={eps:.5f}) accuracy -> TransferResNet: {resnet_adv_acc:.4f}")


if __name__ == "__main__":
    main()
