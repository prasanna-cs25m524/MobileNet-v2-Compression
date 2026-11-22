import os
import argparse
import torch
import torch.optim as optim
import wandb

from dataset import get_dataloaders
from model import get_model
from train import train
from evaluate import evaluate_failure_modes
from test import run_quantization_evaluation
from utils import plot_training_curves, count_parameters, set_seed

WANDB_API_KEY   = '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'
DEFAULT_PROJECT = 'CS6886W-MobileNetV2-Quantization'
DEFAULT_CKPT    = './checkpoints/best_mobilenetv2.pth'


def run_training(epochs=100, batch_size=128, lr=0.05, seed=42,
                 checkpoint_path=DEFAULT_CKPT, project=DEFAULT_PROJECT):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Seed: {seed}")

    api_key = os.environ.get("WANDB_API_KEY", WANDB_API_KEY)
    if api_key:
        wandb.login(key=api_key)

    wandb.init(
        project=project,
        config={
            "architecture": "MobileNet-v2",
            "dataset": "CIFAR-10",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": "nesterov_sgd",
            "seed": seed
        },
        name=f"baseline_mobilenetv2_ep{epochs}_lr{lr}_bs{batch_size}"
    )

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = get_model(num_classes=10).to(device)

    param_info = count_parameters(model)
    print(f"MobileNet-v2 Params: Total={param_info['total']:,}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    history, best_acc = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        use_wandb=True
    )

    plot_training_curves(history, save_path="./data/training_curves.png")

    print(f"\nFinal Best Test Accuracy: {best_acc:.2f}%")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
        model.load_state_dict(state_dict)
        evaluate_failure_modes(model, test_loader, device)

    wandb.summary["best_val_acc"] = best_acc
    wandb.finish()
    return best_acc


def run_sweep(checkpoint_path=DEFAULT_CKPT, project=DEFAULT_PROJECT, calib_batches=20):
    configs = [
        {"w_bits": 8, "a_bits": 8, "per_channel": True,  "desc": "W8A8_PerChannel"},
        {"w_bits": 8, "a_bits": 6, "per_channel": True,  "desc": "W8A6_PerChannel"},
        {"w_bits": 8, "a_bits": 4, "per_channel": True,  "desc": "W8A4_PerChannel"},
        {"w_bits": 8, "a_bits": 8, "per_channel": False, "desc": "W8A8_PerTensor"},

        {"w_bits": 6, "a_bits": 8, "per_channel": True,  "desc": "W6A8_PerChannel"},
        {"w_bits": 6, "a_bits": 6, "per_channel": True,  "desc": "W6A6_PerChannel"},
        {"w_bits": 6, "a_bits": 4, "per_channel": True,  "desc": "W6A4_PerChannel"},
        {"w_bits": 6, "a_bits": 6, "per_channel": False, "desc": "W6A6_PerTensor"},

        {"w_bits": 4, "a_bits": 8, "per_channel": True,  "desc": "W4A8_PerChannel"},
        {"w_bits": 4, "a_bits": 6, "per_channel": True,  "desc": "W4A6_PerChannel"},
        {"w_bits": 4, "a_bits": 4, "per_channel": True,  "desc": "W4A4_PerChannel"},
        {"w_bits": 4, "a_bits": 4, "per_channel": False, "desc": "W4A4_PerTensor"},
    ]

    print(f"Starting Quantization Sweep: {len(configs)} configurations on project '{project}'")

    for idx, cfg in enumerate(configs, start=1):
        print(f"\n--- Run [{idx}/{len(configs)}]: {cfg['desc']} ---")

        run = wandb.init(
            project=project,
            name=f"quant_{cfg['desc']}",
            config={
                "weight_bits": cfg["w_bits"],
                "activation_bits": cfg["a_bits"],
                "per_channel": cfg["per_channel"],
                "calib_batches": calib_batches,
            },
            reinit=True
        )

        res = run_quantization_evaluation(
            weight_bits=cfg["w_bits"],
            act_bits=cfg["a_bits"],
            checkpoint_path=checkpoint_path,
            batch_size=128,
            calib_batches=calib_batches,
            per_channel=cfg["per_channel"]
        )

        wandb.log({
            "weight_bits": cfg["w_bits"],
            "activation_bits": cfg["a_bits"],
            "per_channel": 1 if cfg["per_channel"] else 0,
            "quant_test_acc": res["quant_acc"],
            "acc_drop": res["acc_drop"],
            "model_size_mb": res["model_summary"]["quant_model_mb"],
            "weight_compression_ratio": res["model_summary"]["compression_ratio_weights"],
            "activation_compression_ratio": res["act_summary"]["compression_ratio"],
            "total_model_compression_ratio": res["model_summary"]["compression_ratio_model"],
            "metadata_overhead_kb": res["model_summary"]["total_overhead_kb"]
        })

        run.finish()

    print("\nSweep Complete! View your Parallel Coordinates chart on W&B.")


def main():
    parser = argparse.ArgumentParser(description="MobileNet-v2 CIFAR-10 Training & Quantization")

    # Mode selection
    parser.add_argument('--sweep', action='store_true',
                        help='Run automated W&B quantization sweep (12 configurations)')
    parser.add_argument('--quantize', action='store_true',
                        help='Run a single manual quantization evaluation')

    # Quantization parameters (for manual --quantize run)
    parser.add_argument('--weight_bits', type=int, default=8,
                        help='Weight quantization bits (default: 8)')
    parser.add_argument('--act_bits', type=int, default=8,
                        help='Activation quantization bits (default: 8)')
    parser.add_argument('--no_per_channel', action='store_true',
                        help='Use per-tensor scaling instead of per-channel')
    parser.add_argument('--calib_batches', type=int, default=20,
                        help='Calibration batches (default: 20)')

    # Baseline training parameters (default mode)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')

    # Shared parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT,
                        help=f'Checkpoint path (default: {DEFAULT_CKPT})')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
                        help=f'W&B project name (default: {DEFAULT_PROJECT})')

    args = parser.parse_args()
    set_seed(args.seed)

    api_key = os.environ.get("WANDB_API_KEY", WANDB_API_KEY)
    if api_key:
        wandb.login(key=api_key)

    if args.sweep:
        # Automated W&B Sweep
        run_sweep(
            checkpoint_path=args.checkpoint,
            project=args.project,
            calib_batches=args.calib_batches
        )
    elif args.quantize:
        # Single Manual Quantization Run
        run_quantization_evaluation(
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            calib_batches=args.calib_batches,
            per_channel=not args.no_per_channel
        )
    else:
        # Default: Baseline Model Training
        run_training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            project=args.project
        )


if __name__ == '__main__':
    main()
