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
├── test.py           # Standalone quantization evaluation (matching assignment handout)
├── utils.py          # Forward hook activation memory profiler and plot helpers
├── main.py           # Unified entry point for training, manual eval, and W&B sweep
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

---

## 4. Seed Configuration
Fixed random seed is used across python, numpy, and torch via `set_seed(42)` to ensure reproducibility. You can also pass custom seed via `--seed <int>`.

---

## 5. How to Run

### Option 1: Baseline Training (Default Mode)
Trains MobileNet-v2 on CIFAR-10 (100 epochs, Nesterov SGD, Cosine Annealing, seed 42):
```bash
python main.py
```
*Saves best model to `./checkpoints/best_mobilenetv2.pth` and loss/accuracy plots to `./data/training_curves.png`.*

### Option 2: Manual Quantization Evaluation
Evaluate post-training quantization on the trained checkpoint:
```bash
# Using main.py
python main.py --quantize --weight_bits 8 --act_bits 8

# Or using test.py (matching assignment handout syntax)
python test.py --weight_quant_bits 8 --activation_quant_bits 8
```

### Option 3: Automated Quantization Sweep (Question 3)
Runs the required simulation sweep (12 configurations across 8-bit, 6-bit, and 4-bit weights/activations) and logs results to W&B:
```bash
python main.py --sweep
```
After the sweep completes, add a **Parallel Coordinates** panel in your W&B project dashboard to view the chart.
