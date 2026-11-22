import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 channel statistics (Standard public dataset values)
RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD  = (0.2023, 0.1994, 0.2010)


class Cutout(object):
    """
    Randomly mask out one or more square patches from an image tensor.
    Regularization technique to prevent lazy feature reliance and boost generalization.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with masked-out patch(es).
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img


def get_dataloaders(batch_size=128, data_dir='./data', num_workers=2):
    """
    Prepare CIFAR-10 train and test dataloaders with data augmentation and normalization.
    Augmentation pipeline:
      - RandomCrop(32, padding=4)
      - RandomHorizontalFlip()
      - RandomRotation(5 degrees)
      - ColorJitter(brightness=0.1, contrast=0.1)
      - ToTensor()
      - Normalize(RGB_MEAN, RGB_STD)
      - Cutout(n_holes=1, length=16)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD),
        Cutout(n_holes=1, length=16),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD),
    ])

    train_data = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader
