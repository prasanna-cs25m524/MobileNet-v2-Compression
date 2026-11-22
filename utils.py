import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Measure activation memory using forward hooks
def measure_activation_memory(model, input_shape=(1, 3, 32, 32), act_bits=8, device='cpu'):
    activation_elements = []
    hooks = []

    def hook_fn(module, input_val, output_val):
        if isinstance(output_val, torch.Tensor):
            activation_elements.append((module.__class__.__name__, output_val.numel()))
        elif isinstance(output_val, (tuple, list)):
            for item in output_val:
                if isinstance(item, torch.Tensor):
                    activation_elements.append((module.__class__.__name__, item.numel()))

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    dummy_input = torch.zeros(input_shape, device=device)

    with torch.no_grad():
        _ = model(dummy_input)

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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}


def plot_training_curves(history, save_path="training_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'], label='Val Acc')
    ax2.set_title('Training & Validation Top-1 Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved curves to: {save_path}")
