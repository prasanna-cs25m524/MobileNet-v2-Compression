"""
main.py: Entry point for training baseline MobileNet-v2 on CIFAR-10.
Supports CLI overrides, automated seed setting, W&B logging, and learning rate scheduling.
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import wandb

from dataset import get_dataloaders
from model import get_model
from train import train, get_optimizer
from evaluate import evaluate
from utils import plot_training_curves, count_parameters


# Default Baseline Configuration
DEFAULT_PROJECT    = 'CS6886W-MobileNetV2-Quantization'
DEFAULT_SEED       = 42
DEFAULT_OPTIMIZER  = 'nesterov_sgd'
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR         = 0.05
DEFAULT_EPOCHS     = 100
DEFAULT_WIDTH_MULT = 1.0
DEFAULT_DROPOUT    = 0.2
DEFAULT_CKPT_PATH  = './checkpoints/best_mobilenetv2.pth'


def set_seed(seed=42):
    """Fix random seeds across python, numpy, torch, and CUDA for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_experiment(manual_config=None, project=DEFAULT_PROJECT):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg_dict = manual_config or {
        "seed": DEFAULT_SEED,
        "optimizer": DEFAULT_OPTIMIZER,
        "lr": DEFAULT_LR,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "width_mult": DEFAULT_WIDTH_MULT,
        "dropout": DEFAULT_DROPOUT,
    }

    # Initialize W&B if not already active
    if wandb.run is None:
        wandb.init(
            project=project,
            config=cfg_dict,
            name=f"baseline_mobilenetv2_{cfg_dict['optimizer']}_lr{cfg_dict['lr']}_ep{cfg_dict['epochs']}"
        )
        config = wandb.config
    else:
        config = wandb.config

    # Set seed
    set_seed(config.seed)

    # Dataloaders
    train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)

    # Model
    model = get_model(num_classes=10, width_mult=config.width_mult, dropout=config.dropout)
    model.to(device)

    param_info = count_parameters(model)
    print(f"MobileNet-v2 Parameters: Total={param_info['total']:,} | Trainable={param_info['trainable']:,}")

    # Optimizer & Cosine Annealing LR Scheduler
    optimizer = get_optimizer(config.optimizer, model, lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-4)

    # Train
    history, best_acc = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config.epochs,
        checkpoint_path=DEFAULT_CKPT_PATH,
        use_wandb=True
    )

    # Plot & save curves (Question 1c)
    plot_training_curves(history, save_path="./data/training_curves.png")

    print(f"\nFinal Baseline Model Top-1 Accuracy: {best_acc:.2f}%")
    wandb.summary["best_val_acc"] = best_acc
    wandb.finish()
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="MobileNet-v2 on CIFAR-10 Training")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
                        help=f'W&B project name (default: {DEFAULT_PROJECT})')
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER,
                        choices=['sgd', 'nesterov_sgd', 'adam', 'adamw', 'adagrad', 'rmsprop', 'nadam'],
                        help=f'Optimizer (default: {DEFAULT_OPTIMIZER})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'Learning rate (default: {DEFAULT_LR})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--width_mult', type=float, default=DEFAULT_WIDTH_MULT,
                        help=f'Width multiplier alpha (default: {DEFAULT_WIDTH_MULT})')
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT,
                        help=f'Dropout rate (default: {DEFAULT_DROPOUT})')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluate existing checkpoint without training')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT_PATH,
                        help=f'Path to checkpoint file (default: {DEFAULT_CKPT_PATH})')

    args = parser.parse_args()
    set_seed(args.seed)

    # Automated W&B Login
    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        wandb.login(key=api_key)
    else:
        try:
            wandb.login()
        except Exception:
            print("W&B login skipped or interactive prompt available.")

    if args.eval_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, test_loader = get_dataloaders(batch_size=args.batch_size)
        model = get_model(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from: {args.checkpoint}")
        else:
            print(f"Error: checkpoint {args.checkpoint} not found.")
            sys.exit(1)
        model.to(device)
        acc = evaluate(model, test_loader, device)
        print(f"Baseline Test Accuracy: {acc:.2f}%")
    else:
        manual_config = {
            "seed": args.seed,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "width_mult": args.width_mult,
            "dropout": args.dropout,
        }
        run_experiment(manual_config=manual_config, project=args.project)


if __name__ == '__main__':
    main()
