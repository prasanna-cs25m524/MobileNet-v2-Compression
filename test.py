import os
import argparse
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import get_model
from evaluate import evaluate
from quantize import (
    swap_to_quant_modules,
    freeze_all_quant,
    compute_model_in_memory_size,
    print_compression_report
)
from utils import measure_activation_memory, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_quantization_evaluation(weight_bits=8, act_bits=8,
                                checkpoint_path='./checkpoints/best_mobilenetv2.pth',
                                batch_size=128, calib_batches=20,
                                per_channel=True, width_mult=1.0, dropout=0.2):
    print("=" * 60)
    print(f"Quantization Eval: Weights={weight_bits}-bit, Activations={act_bits}-bit")
    print(f"Device: {device} | Per-Channel: {per_channel}")
    print("=" * 60)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = get_model(num_classes=10, width_mult=width_mult, dropout=dropout)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found.")

    model.to(device)
    model.eval()

    # Baseline accuracy
    baseline_acc = evaluate(model, test_loader, device)
    print(f"Baseline FP32 Test Acc: {baseline_acc:.2f}%")

    # Swap to custom quantization modules
    swap_to_quant_modules(
        model,
        weight_bits=weight_bits,
        act_bits=act_bits,
        per_channel=per_channel,
        activations_unsigned=True
    )
    model.to(device)

    # Calibration on training batches
    print(f"Calibrating activation stats over {calib_batches} batches...")
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            _ = model(images)
            if (i + 1) >= calib_batches:
                break

    # Freeze quant parameters
    freeze_all_quant(model)

    # Quantized model accuracy
    quant_acc = evaluate(model, test_loader, device)
    acc_drop = baseline_acc - quant_acc
    print(f"Quantized Test Acc: {quant_acc:.2f}% (Drop: {acc_drop:+.2f}%)")

    # Compute in-memory footprint and storage overhead
    model_summary = compute_model_in_memory_size(
        model, weight_bits=weight_bits, per_channel=per_channel, act_bits=act_bits
    )

    # Measure activation footprint using forward hooks
    act_summary = measure_activation_memory(
        model, input_shape=(1, 3, 32, 32), act_bits=act_bits, device=device
    )

    print_compression_report(model_summary, act_summary)

    return {
        "baseline_acc": baseline_acc,
        "quant_acc": quant_acc,
        "acc_drop": acc_drop,
        "model_summary": model_summary,
        "act_summary": act_summary
    }


def main():
    parser = argparse.ArgumentParser(description="Quantization Evaluation")
    parser.add_argument('--weight_quant_bits', type=int, default=8,
                        help='Weight bits (default: 8)')
    parser.add_argument('--activation_quant_bits', type=int, default=8,
                        help='Activation bits (default: 8)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_mobilenetv2.pth',
                        help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--calib_batches', type=int, default=20)
    parser.add_argument('--no_per_channel', action='store_true',
                        help='Use per-tensor instead of per-channel')
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)

    run_quantization_evaluation(
        weight_bits=args.weight_quant_bits,
        act_bits=args.activation_quant_bits,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        calib_batches=args.calib_batches,
        per_channel=not args.no_per_channel,
        width_mult=args.width_mult,
        dropout=args.dropout
    )


if __name__ == '__main__':
    main()
