"""
Question 5: CIFAR-10 dataloaders with train/val/test splits.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Data transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_dataloaders(batch_size, data_dir=None, seed=42, num_workers=2):
    """
    Download CIFAR-10, split train into 80/20 train/val, and return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # Full training set (with train_transform applied)
    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # Validation set uses test_transform (no augmentation)
    full_train_for_val = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform
    )

    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    # Use the same indices for validation but with test_transform
    val_subset = torch.utils.data.Subset(full_train_for_val, val_subset.indices)

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
