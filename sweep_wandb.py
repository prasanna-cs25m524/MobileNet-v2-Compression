import os
import argparse
import torch
import wandb

from test import run_quantization_evaluation

DEFAULT_PROJECT = 'CS6886W-MobileNetV2-Quantization'


def run_sweep(checkpoint_path='./checkpoints/best_mobilenetv2.pth',
              project=DEFAULT_PROJECT, calib_batches=20):
    # 12 configurations across 8, 6, 4 bits and per-channel vs per-tensor
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

    print(f"Starting sweep: {len(configs)} configurations")

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

    print("\nSweep finished! You can view the Parallel Coordinates chart on W&B.")


def main():
    parser = argparse.ArgumentParser(description="W&B Quantization Sweep")
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_mobilenetv2.pth')
    parser.add_argument('--calib_batches', type=int, default=20)

    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        wandb.login(key=api_key)
    else:
        try:
            wandb.login()
        except Exception:
            pass

    run_sweep(
        checkpoint_path=args.checkpoint,
        project=args.project,
        calib_batches=args.calib_batches
    )


if __name__ == '__main__':
    main()
