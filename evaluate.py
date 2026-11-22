# evaluate.py
# Responsible for: computing top-1 accuracy on a dataloader

import torch


def evaluate(model, loader, device):
    """
    Computes top-1 classification accuracy on the provided dataloader.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / max(total, 1)
    return accuracy
