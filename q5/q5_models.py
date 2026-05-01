"""
Question 5: Model architectures (DeepMLP, CustomCNN, TransferResNet).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DeepMLP(nn.Module):
    """
    MLP with 4 hidden layers of 512 units each.
    Each hidden layer: Linear -> BatchNorm1d -> ReLU -> Dropout.
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        input_dim = 32 * 32 * 3
        hidden_dim = 512

        for i in range(4):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


class CustomCNN(nn.Module):
    """
    CNN with 3 conv blocks and a final linear classifier.
    Each block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout),
        )

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class TransferResNet(nn.Module):
    """
    ResNet18 with frozen layers except layer4 and fc.
    """

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
