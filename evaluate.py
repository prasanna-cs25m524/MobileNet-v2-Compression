# evaluate.py
# Responsible for: computing top-1 accuracy on a dataloader

import torch

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


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


def evaluate_failure_modes(model, loader, device):
    """
    Computes per-class accuracy and confusion pairs for Question 1(c) failure mode analysis.
    """
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1

    per_class_acc = {}
    print("\n--- Per-Class Top-1 Accuracy (Failure Mode Analysis) ---")
    for i in range(10):
        acc = 100.0 * class_correct[i] / max(class_total[i], 1)
        per_class_acc[CIFAR10_CLASSES[i]] = acc
        print(f"  {CIFAR10_CLASSES[i]:12s}: {acc:.2f}%")

    # Find top confused pairs (excluding diagonal)
    conf_diag_zero = confusion_matrix.clone()
    for i in range(10):
        conf_diag_zero[i, i] = 0
    max_confusions = []
    for _ in range(3):
        idx = torch.argmax(conf_diag_zero)
        true_cls = (idx // 10).item()
        pred_cls = (idx % 10).item()
        count = conf_diag_zero[true_cls, pred_cls].item()
        max_confusions.append((CIFAR10_CLASSES[true_cls], CIFAR10_CLASSES[pred_cls], count))
        conf_diag_zero[true_cls, pred_cls] = 0

    print("\nTop Misclassified Class Pairs:")
    for true_c, pred_c, cnt in max_confusions:
        print(f"  {true_c} misclassified as {pred_c}: {cnt} times")

    return per_class_acc, max_confusions
