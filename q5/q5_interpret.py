"""
Question 5: Interpretability with Grad-CAM and SHAP.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam, LayerAttribution
import shap

# Outputs are organized in outputs/q5/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q5')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_misclassified(model, dataloader, device=None, max_samples=10):
    """
    Find misclassified samples from the dataloader.

    Returns:
        images: tensor (N, C, H, W)
        true_labels: tensor (N,)
        pred_labels: tensor (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    images_list = []
    true_list = []
    pred_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            mismatches = preds != targets

            if mismatches.any():
                idx = torch.where(mismatches)[0]
                for i in idx:
                    images_list.append(inputs[i].cpu())
                    true_list.append(targets[i].cpu())
                    pred_list.append(preds[i].cpu())

                    if len(images_list) >= max_samples:
                        break

            if len(images_list) >= max_samples:
                break

    if len(images_list) == 0:
        return None, None, None

    images = torch.stack(images_list)
    true_labels = torch.stack(true_list)
    pred_labels = torch.stack(pred_list)

    return images, true_labels, pred_labels


def plot_gradcam_misclassified(model, dataloader, class_names=None, device=None, save_path=None):
    """
    Generate Grad-CAM heatmaps for 10 misclassified images using last conv layer.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'cnn_gradcam_failures.jpg')

    images, true_labels, pred_labels = find_misclassified(model, dataloader, device=device, max_samples=10)
    if images is None:
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    images = images.to(device)

    # Assume last conv layer is model.block3[0] (Conv2d) for CustomCNN
    last_conv = model.block3[0]
    gradcam = LayerGradCam(model, last_conv)

    n = images.size(0)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2 * n))

    for i in range(n):
        img = images[i:i+1]
        pred = pred_labels[i].item()
        true = true_labels[i].item()

        # Grad-CAM attribution
        attribution = gradcam.attribute(img, target=pred)
        attribution = LayerAttribution.interpolate(attribution, img.shape[-2:])
        heatmap = attribution.squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Original image
        img_np = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np * 0.5) + 0.5  # unnormalize
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        label_true = class_names[true] if class_names else str(true)
        label_pred = class_names[pred] if class_names else str(pred)
        axes[i, 0].set_title(f"True: {label_true} | Pred: {label_pred}")

        axes[i, 1].imshow(img_np)
        axes[i, 1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Grad-CAM")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mlp_shap(model, dataloader, class_names=None, device=None, save_path=None):
    """
    Apply SHAP on DeepMLP for 3 misclassified images and plot attributions.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'mlp_shap.jpg')

    images, true_labels, pred_labels = find_misclassified(model, dataloader, device=device, max_samples=3)
    if images is None:
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # SHAP expects 2D inputs for MLP
    images_flat = images.view(images.size(0), -1)

    # Background for SHAP
    background = images_flat[:1].to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(images_flat.to(device))

    # Plot SHAP attributions
    fig, axes = plt.subplots(3, 2, figsize=(6, 9))

    for i in range(3):
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5
        img_np = np.clip(img_np, 0, 1)

        # Select SHAP values for the predicted class
        pred_class = pred_labels[i].item()
        if isinstance(shap_values, list):
            shap_vec = shap_values[pred_class][i]
        else:
            shap_vec = shap_values[i]

        shap_flat = np.abs(shap_vec).reshape(-1)
        n_features = 32 * 32 * 3
        if shap_flat.size != n_features and shap_flat.size % n_features == 0:
            shap_flat = shap_flat.reshape(-1, n_features)
            class_idx = min(pred_class, shap_flat.shape[0] - 1)
            shap_flat = shap_flat[class_idx]
        shap_map = shap_flat.reshape(32, 32, 3)
        shap_map = shap_map.mean(axis=2)

        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        label_true = class_names[true_labels[i].item()] if class_names else str(true_labels[i].item())
        label_pred = class_names[pred_labels[i].item()] if class_names else str(pred_labels[i].item())
        axes[i, 0].set_title(f"True: {label_true} | Pred: {label_pred}")

        axes[i, 1].imshow(shap_map, cmap='coolwarm')
        axes[i, 1].axis('off')
        axes[i, 1].set_title("SHAP Attribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
