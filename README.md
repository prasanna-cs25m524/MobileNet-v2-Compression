# **CS6886W: Systems Engineering for Deep Learning** - Assignment 2

**Department of Computer Science and Engineering, IIT Madras**  
**Name**: Prasanna JS  
**Roll Number**: CS25M524  

---

## 1. Overview
In this assignment, we train **MobileNet-v2** on the **CIFAR-10** dataset and then apply custom quantization on both weights and activations to reduce the in-memory footprint (RAM/VRAM) while retaining accuracy.

No external compression APIs (like `torch.ao.quantization` or `bitsandbytes`) are used. All quantization and calibration logic is written from scratch.

---

## 2. Code Structure
```text
mobilenetv2_compression/
├── dataset.py        # CIFAR-10 loading, augmentations (Cutout, Crop, Flip, Jitter)
├── model.py          # MobileNet-v2 adapted for CIFAR-10 (stride=1 first conv)
├── train.py          # Training loop with Cosine Annealing and W&B logging
├── evaluate.py       # Top-1 accuracy evaluation
├── quantize.py       # Custom quantization (weights & activations, per-channel/per-tensor)
├── test.py           # Evaluation script for quantized model
├── sweep_wandb.py    # W&B sweep script for Parallel Coordinates plot (>= 8 runs)
├── utils.py          # Forward hook activation memory profiler and plot helpers
├── main.py           # Main training entry point
└── README.md         # Project documentation
```

---

## 3. Environment Setup & Dependencies
- Python: 3.10+
- Hardware: GPU with CUDA (tested on Google Colab)

Install requirements:
```bash
pip install torch torchvision wandb matplotlib
```

### W&B Setup
Set your API key before running or login when prompted:
```bash
export WANDB_API_KEY="YOUR_KEY"
```

---

## 4. Seed Configuration
Fixed random seed is used across python, numpy, and torch via `set_seed(42)` to ensure reproducibility. You can also pass custom seed via `--seed <int>`.

---

## 5. How to Run

### Step 1: Train Baseline Model
Trains MobileNet-v2 on CIFAR-10 for 100 epochs using Nesterov SGD, cosine annealing, and data augmentation:
```bash
python main.py --epochs 100 --batch_size 128 --lr 0.05 --optimizer nesterov_sgd --seed 42
```
*Saves best model to `./checkpoints/best_mobilenetv2.pth` and loss/accuracy plots to `./data/training_curves.png`.*

### Step 2: Test Quantization
Run post-training quantization on the trained checkpoint:
```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8 --checkpoint checkpoints/best_mobilenetv2.pth
```
You can vary the bits, for example 4-bit weights and 4-bit activations:
```bash
python test.py --weight_quant_bits 4 --activation_quant_bits 4 --checkpoint checkpoints/best_mobilenetv2.pth
```

### Step 3: Run W&B Sweep for Parallel Coordinates Chart
Runs the required simulation sweep (12 configurations across 8-bit, 6-bit, and 4-bit weights/activations) and logs results to W&B:
```bash
python sweep_wandb.py --checkpoint checkpoints/best_mobilenetv2.pth
```
After the sweep completes, add a **Parallel Coordinates** panel in your W&B project dashboard to view the chart.
