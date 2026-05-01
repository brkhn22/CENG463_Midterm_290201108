"""
Question 5: Adversarial robustness testing with FGSM (torchattacks).
"""

import torch
import torchattacks


def fgsm_attack_accuracy(model, test_loader, eps=8/255, device=None):
    """
    Apply FGSM attack and evaluate accuracy on adversarial examples.

    Returns:
        adv_accuracy: float
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    attack = torchattacks.FGSM(model, eps=eps)

    total = 0
    correct = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        adv_inputs = attack(inputs, targets)
        outputs = model(adv_inputs)
        preds = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += (preds == targets).sum().item()

    adv_accuracy = correct / total if total > 0 else 0.0
    return adv_accuracy
