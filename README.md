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
├── data/
│   └── runlogs/      # Training and evaluation logs
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
# Default baseline (100 epochs):
python main.py

# Baseline with a custom W&B run name / project:
python main.py --run_name "baseline_mobilenetv2_100ep_clean"
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

---

## 6. Baseline Configuration

### MobileNet-v2 Architecture:
- **Total Parameters:** 2,236,682 (Trainable: 2,236,682)
- **Width Multiplier ($\alpha$):** `1.0`
- **Activation Function:** `ReLU6` (upper-bounded at 6 to preserve dynamic range under low-bit quantization)
- **Channel Divisibility:** `round_nearest=8` via `_make_divisible()` (ensures hardware-aligned channel widths)
- **Dropout:** `0.2` (applied before the final linear classifier)
- **Batch Normalization:** `BatchNorm2d` after each convolution (`momentum=0.1`, `eps=1e-5`)
- **CIFAR-10 Adaptation:** Initial Conv2d uses `stride=1` (instead of `stride=2` for ImageNet) to preserve spatial dimensions on $32\times 32$ inputs

#### Layer-by-Layer Block Breakdown:
The backbone consists of an initial standard convolution, followed by **17 Inverted Residual bottleneck blocks** organized across 7 stages, a 1x1 point-wise expansion convolution, global average pooling, and a linear classifier:

| Stage | Layer / Operator | Input Size | Expansion ($t$) | Output Channels ($c$) | Repeats ($n$) | Stride ($s$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Input | `Conv2d 3x3 + BN + ReLU6` | $32 \times 32 \times 3$ | — | 32 | 1 | 1 |
| 1 | `InvertedResidual` | $32 \times 32 \times 32$ | 1 | 16 | 1 | 1 |
| 2 | `InvertedResidual` | $32 \times 32 \times 16$ | 6 | 24 | 2 | 1 |
| 3 | `InvertedResidual` | $32 \times 32 \times 24$ | 6 | 32 | 3 | 2 |
| 4 | `InvertedResidual` | $16 \times 16 \times 32$ | 6 | 64 | 4 | 2 |
| 5 | `InvertedResidual` | $8 \times 8 \times 64$ | 6 | 96 | 3 | 1 |
| 6 | `InvertedResidual` | $8 \times 8 \times 96$ | 6 | 160 | 3 | 2 |
| 7 | `InvertedResidual` | $4 \times 4 \times 160$ | 6 | 320 | 1 | 1 |
| Head | `Conv2d 1x1 + BN + ReLU6` | $4 \times 4 \times 320$ | — | 1280 | 1 | 1 |
| Pool | `Global AvgPool2d` | $4 \times 4 \times 1280$ | — | 1280 | 1 | — |
| Out | `Dropout(0.2) + Linear` | $1 \times 1 \times 1280$ | — | 10 | 1 | — |

*Each Inverted Residual block consists of: $1\times 1$ Expansion Conv ($\text{ReLU6}$) $\rightarrow$ $3\times 3$ Depthwise Conv ($\text{ReLU6}$) $\rightarrow$ $1\times 1$ Linear Projection Conv ($\text{no activation}$), with a residual skip connection when $s=1$ and $c_{\text{in}} = c_{\text{out}}$.*

### Data Pipeline & Augmentation:
- **Normalization:** Per-channel mean `(0.4914, 0.4822, 0.4465)` and std `(0.2023, 0.1994, 0.2010)`
- **Augmentation Pipeline:**
  - `RandomCrop(32, padding=4)`: In the dataset, photos naturally capture the object at the center of the frame so there is a possibility that it may learn or memorize the center coordinates. To avoid that, we randomly cropped the image.
  - `RandomHorizontalFlip()`: Sometimes the image can be facing left-to-right or right-to-left (e.g. car facing left and car facing right should both be recognized as car), so we flip the image horizontally.
  - `RandomRotation(degrees=5)`: In photos we can take pictures in any orientation or tilt, so added up to 5-degree tilt to the image.
  - `ColorJitter(brightness=0.1, contrast=0.1)`: If all photos of a certain object (e.g. car) have similar brightness, the ANN may memorize bright photo = car, so we add random brightness and contrast.
  - `Cutout(n_holes=1, length=16)`: The ANN can learn one feature dominantly (for example when recognizing a bird, if all images have clear eyes and beak, it may not learn wings or feathers). Dropping a random 16x16 black square over photos forces it to learn all features.
- **Batch Size:** `128`

### Training Strategy:
- **Optimizer:** `SGD` with `momentum=0.9`, `nesterov=True`, and `weight_decay=5e-4`
- **Initial Learning Rate:** `0.05`
- **LR Schedule:** `CosineAnnealingLR` ($T_{\max}=100$, $\eta_{\min}=10^{-4}$)
- **Loss Function:** `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Epochs:** `100`
- **Random Seed:** `42` for o/p reproducibility

---

## 7. Baseline Results

The baseline MobileNet-v2 was trained for 100 epochs with random seed 42.

- **Best Test Accuracy:** **95.00%** (Epoch 100)
- **Train Accuracy:** 95.94%
- **Train Loss:** 0.5964
- **Val Loss:** 0.6231
- **Checkpoint:** Saved to `./checkpoints/best_mobilenetv2.pth` (generated upon training)
- **Reference Run Log:** Full console output from the 100-epoch baseline training run is provided at [`data/runlogs/baseline_train_run_log.txt`](https://github.com/prasanna-cs25m524/MobileNet-v2-Compression/blob/main/data/runlogs/baseline_train_run_log.txt)

### Per-Class Test Accuracy:

| Class | Accuracy | Notes / Failure Modes |
| :--- | :---: | :--- |
| Automobile | 97.90% | Misunderstood as truck 18 times |
| Ship | 97.20% | Clean separation, water background helps |
| Truck | 97.20% | High accuracy, few confused with cars |
| Frog | 96.50% | Good separation |
| Horse | 96.10% | Good separation |
| Deer | 95.80% | Minor confusion with horse/bird |
| Airplane | 95.40% | High accuracy, sky background helps |
| Bird | 93.90% | Some confusion with airplanes/small animals |
| Dog | 91.50% | Frequently predicted as cat (57 times) |
| Cat | 88.50% | Lowest accuracy; predicted as dog 68 times |

**Observations:**
- Cat detection test accuracy is lowest at 88.50%. So I would say the biggest error was between cats and dogs because 68 cats were predicted as dogs, and 57 dogs were predicted as cats. This one confusion is equivalent to more than half of all misclassified images.
- Maybe I am thinking in 32x32 resolution, cats and dogs look similar making them easy to confuse, so if we try with 224x224 we would have not seen this drop.
- Few of automobiles were misunderstood as trucks (18 times) because maybe there is confusion between wheels, road backgrounds, etc., as that will mostly look the same in 32x32 resolution.

---

## 8. Model Compression Implementation

### 1. Configurable Compression & Design Choices:
- Custom quantization written in `quantize.py` without external compression libraries. Configurable for 8-bit, 6-bit, and 4-bit for both weights and activations.
- **Weights:**
  - Symmetric signed quantization with Zero Point = 0. Weights are already centered around zero, so setting Zero Point = 0 saves RAM (no zero-point values stored) and avoids subtracting zero-points during multiplication.
  - Per-channel scaling (one scale per output channel) instead of one scale for the whole layer. MobileNet-v2 uses depthwise convolutions where different channels have very different weight ranges. Per-tensor scaling rounded small channels to zero and destroyed accuracy; per-channel scaling fixed this completely.
- **Activations:**
  - Asymmetric unsigned quantization ($0$ to $255$ for 8-bit). MobileNet-v2 uses `ReLU6`, so activations are always positive ($0$ to $6$). Using unsigned gives all 256 buckets for the positive range instead of wasting half on negative numbers that never occur.
  - Calibrated over 20 training batches to find and freeze min/max activation values, providing stable scaling factors before testing.

### 2. How Compression is Applied to MobileNet-v2:
- MobileNet-v2 has layers nested inside 17 `InvertedResidual` blocks, so a simple sequential loop could not find them. Wrote a recursive function `swap_to_quant_modules()` that goes through all child blocks and swaps:
  - `nn.Conv2d` $\rightarrow$ `QuantConv2d`
  - `nn.Linear` $\rightarrow$ `QuantLinear`
  - `nn.ReLU6` $\rightarrow$ wrapped with `ActFakeQuant` (so as soon as activations come out of `ReLU6`, they are calibrated and rounded to discrete 8-bit / 6-bit / 4-bit integer levels before entering the next convolution layer).
- **Layers Compressed:**
  - The first $3\times 3$ input conv.
  - All 17 Inverted Residual blocks ($1\times 1$ expansion, $3\times 3$ depthwise, and $1\times 1$ projection convs).
  - The final $1\times 1$ conv.
  - The final Linear classification layer.
  - All 51 intermediate activation outputs.
- **Layers Not Touched (Exceptions):**
  - Biases kept as normal float32 because summing many 8-bit multiplications can easily overflow if the bias is also low-bit.
  - BatchNorm not quantized separately because its parameters are linear and merged directly into the convolution weights at test time.

### 3. Storage Overheads & In-Memory Size Estimates:
Real in-memory RAM usage is calculated by adding quantized weights, float32 biases, and scale metadata stored in RAM:

- **Metadata Overhead:**
  - **Weight scales:** For per-channel, one 32-bit float scale (4 bytes) is stored per output channel ($34,090 \text{ channels} \times 4 \text{ bytes} \approx \mathbf{133.16\text{ KB}}$).
  - **Weight zero-points:** $\mathbf{0\text{ KB}}$ (since Zero Point = 0).
  - **Activation scales & zero-points:** $51 \text{ layers} \times 8 \text{ bytes} \approx \mathbf{0.40\text{ KB}}$.
  - **Total Overhead in RAM:** $133.16 + 0.40 = \mathbf{133.56\text{ KB}}$ ($\approx 0.13\text{ MB}$).

- **In-Memory Size & Results (Weight 8-bit, Activation 8-bit Per-Channel):**

| Component | Uncompressed FP32 | Quantized (W8A8 Per-Channel) | Compression Ratio |
| :--- | :--- | :--- | :---: |
| **Weights** | 8.467 MB | 2.117 MB | **3.77x** |
| **Biases** | 0.065 MB | 0.065 MB (kept FP32) | 1.00x |
| **Metadata (scales)** | 0.000 MB | 0.130 MB (133.56 KB) | — |
| **Total Model in RAM** | **8.532 MB** | **2.312 MB** | **3.69x** |
| **Activations (1 image)** | **24,553 KB** (23.98 MB) | **6,138 KB** (6.00 MB) | **4.00x** |

- **Quantized Test Accuracy:** **94.99%** (Baseline: 95.00%, Accuracy Drop: **0.01%**).
