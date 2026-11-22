import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from evaluate import evaluate


def get_optimizer(name, model, lr=0.05, weight_decay=5e-4):
    """
    Instantiates the requested optimizer.
    Supported: ['sgd', 'nesterov_sgd', 'adam', 'adamw', 'adagrad', 'rmsprop', 'nadam']
    """
    name = name.lower()
    params = model.parameters()

    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'nesterov_sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'nadam':
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: '{name}'. "
                         f"Choose from: ['sgd', 'nesterov_sgd', 'adam', 'adamw', 'adagrad', 'rmsprop', 'nadam']")


def train(model, train_loader, test_loader, optimizer, device, epochs=100,
          scheduler=None, checkpoint_path='./checkpoints/best_mobilenetv2.pth',
          use_wandb=True):
    """
    Executes training of MobileNetV2 with evaluation after each epoch,
    saves the checkpoint with highest test accuracy, and logs metrics to W&B.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    if checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    print(f"\n--- Starting Training: {epochs} Epochs on {device} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        # Epoch metrics
        train_acc = 100.0 * correct / max(total, 1)
        avg_train_loss = running_loss / len(train_loader)

        # Validation metrics
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v_images, v_labels in test_loader:
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                v_outputs = model(v_images)
                val_loss_sum += criterion(v_outputs, v_labels).item()
                _, v_pred = v_outputs.max(1)
                val_total += v_labels.size(0)
                val_correct += v_pred.eq(v_labels).sum().item()

        val_loss = val_loss_sum / len(test_loader)
        test_acc = 100.0 * val_correct / max(val_total, 1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(test_acc)

        # Log metrics to Weights & Biases
        if use_wandb and wandb.run is not None:
            wandb.log({
                "epoch": int(epoch),
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(test_acc),
                "lr": float(optimizer.param_groups[0]['lr'])
            }, commit=True)

        # Checkpoint best model
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

        best_mark = " [*BEST*]" if is_best else ""
        print(f"Epoch [{epoch:03d}/{epochs:03d}]  "
              f"Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%{best_mark}")

    print(f"\nTraining Complete. Best Validation Top-1 Accuracy: {best_acc:.2f}%")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Best checkpoint saved at: {checkpoint_path}")

    return history, best_acc
