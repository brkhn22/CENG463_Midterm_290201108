"""
Question 5: Training loop with early stopping and Optuna tuning.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import optuna

from q5_models import CustomCNN


def _accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_model(model, train_loader, val_loader, optimizer, criterion,
                device=None, max_epochs=15, patience=5):
    """
    Train a model with early stopping.

    Returns:
        train_loss_list, val_loss_list, train_acc_list, val_acc_list
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_state = None
    best_val_acc = -np.inf
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        train_accs = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(_accuracy(outputs, targets))

        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_accs.append(_accuracy(outputs, targets))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_acc = float(np.mean(train_accs))
        val_acc = float(np.mean(val_accs))

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list


def objective(trial, train_loader, val_loader, device=None):
    """
    Optuna objective for tuning CustomCNN.
    Returns best validation accuracy.
    """
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.2, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    model = CustomCNN(dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    _, _, _, val_acc_list = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        max_epochs=15,
        patience=5
    )

    best_val_acc = max(val_acc_list) if val_acc_list else 0.0
    return best_val_acc
