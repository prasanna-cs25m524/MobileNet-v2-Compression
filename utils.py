"""
Utility functions for:
1. Measuring in-memory activation footprint during forward pass using forward hooks.
2. In-memory parameter and storage calculation.
3. Plotting training/validation loss and accuracy curves.
4. Checkpoint saving and loading.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def measure_activation_memory(model, input_shape=(1, 3, 32, 32), act_bits=8, device='cpu'):
    """
    Measures the in-memory activation footprint during an inference forward pass.
    Methodology:
      - Attaches PyTorch forward hooks to all intermediate feature-producing layers
        (Conv2d, BatchNorm2d, ReLU6/ActFakeQuant, Linear, and residual additions).
      - Tracks tensor output shapes and total elements produced in a forward pass.
      - Calculates FP32 RAM footprint (4 bytes/element) vs. Quantized footprint (act_bits/8 bytes/element).
    Returns a dictionary with total elements, FP32 memory (KB/MB), Quantized memory (KB/MB),
    and the activation compression ratio.
    """
    activation_elements = []
    hooks = []

    def hook_fn(module, input_val, output_val):
        if isinstance(output_val, torch.Tensor):
            activation_elements.append((module.__class__.__name__, output_val.numel()))
        elif isinstance(output_val, (tuple, list)):
            for item in output_val:
                if isinstance(item, torch.Tensor):
                    activation_elements.append((module.__class__.__name__, item.numel()))

    # Register forward hooks on intermediate layers
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    was_training = model.training
    dummy_input = torch.zeros(input_shape, device=device)

    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    total_elements = sum(count for _, count in activation_elements)
    fp32_bytes = total_elements * 4.0
    quant_bytes = total_elements * (act_bits / 8.0)
    cr_activations = fp32_bytes / max(quant_bytes, 1.0)

    results = {
        "batch_size": input_shape[0],
        "total_elements": total_elements,
        "fp32_kb": fp32_bytes / 1024.0,
        "fp32_mb": fp32_bytes / (1024.0 * 1024.0),
        "quant_kb": quant_bytes / 1024.0,
        "quant_mb": quant_bytes / (1024.0 * 1024.0),
        "act_bits": act_bits,
        "compression_ratio": cr_activations,
        "layer_breakdown": activation_elements
    }

    return results


def count_parameters(model):
    """Returns total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}


def plot_training_curves(history, save_path="training_curves.png"):
    """
    Plots and saves training & validation loss and accuracy curves (Question 1c).
    history: dict with keys ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy Curve
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'g--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Top-1 Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")
