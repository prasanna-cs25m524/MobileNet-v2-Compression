# **CS6886W: Systems Engineering for Deep Learning** - Assignment 2

**Department of Computer Science and Engineering, IIT Madras**  
**Semester**: Fall 2026  
**Name**: Prasanna JS  
**Roll Number**: CS25M524  

---

## Section 1: Project Overview & Core Requirements

In this assignment, we explore **in-memory model compression** of the **MobileNet-v2** architecture trained on the **CIFAR-10** dataset. The primary objective is to implement **custom weight and activation quantization from scratch** without utilizing external compression APIs (e.g., `torch.ao.quantization`, `torch.quantization`, `bitsandbytes`, or `nncf`).

### Key Assignment Constraints
1. **In-Memory Size (RAM/VRAM), NOT Disk Size**:
   - Compression is strictly evaluated based on the model's footprint when loaded into memory.
   - File-size reduction methods (e.g., Huffman coding, gzip) do not reduce active runtime memory and are not considered.
2. **Custom Implementation**:
   - All quantization math (uniform linear quantization, per-channel/per-tensor scale & zero-point calculation, fake-quantization with dynamic calibration and freezing) is written in pure PyTorch.
3. **Comprehensive Evaluation**:
   - Baseline FP32 accuracy, loss/accuracy curves, failure mode analysis.
   - Storage overhead calculations (scaling factors, zero points, biases).
   - Weights & Biases (W&B) Parallel Coordinates plot across $\ge 8$ simulation configurations.
   - Activation footprint measurement via PyTorch forward hooks.

---

## Section 2: Code Tree Structure

```text
mobilenetv2_compression/
├── .gitignore               # Ignores datasets, checkpoints, W&B logs, cache
├── README.md                # Comprehensive project documentation & reproduction guide
├── Colab_Assignment2.ipynb  # Interactive Google Colab notebook integrated with Google Drive
├── dataset.py               # CIFAR-10 dataloaders, normalization & data augmentations (Cutout)
├── model.py                 # MobileNet-v2 architecture adapted for CIFAR-10 (stride=1, configurable alpha & dropout)
├── evaluate.py              # Top-1 classification accuracy evaluation loop
├── quantize.py              # Custom quantization engine (weights & activations, per-channel/per-tensor, overhead tracking)
├── utils.py                 # Forward hook activation memory profiler, curve plotting, parameter counter
├── train.py                 # Training loop with Cosine Annealing scheduler, checkpointing, and W&B logging
├── main.py                  # CLI entry point to train baseline MobileNet-v2 and log curves
├── test.py                  # Standalone evaluation & post-training quantization script
└── sweep_wandb.py           # Automated grid sweep (12 configurations) generating W&B Parallel Coordinates chart
```

---

## Section 3: Environment Setup & Dependencies

- **Python**: 3.10+
- **Hardware**: NVIDIA GPU with CUDA support (executed on Google Colab GPU / Kaggle).
- **Core Libraries**: PyTorch, Torchvision, Weights & Biases (`wandb`), Matplotlib.

To install dependencies:
```bash
pip install torch torchvision wandb matplotlib
```

### Weights & Biases (W&B) Setup & Auto-Login
To enable live metric tracking and automated chart generation:
1. Export your API key in terminal or Colab:
   ```bash
   export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
   ```
2. Or let `wandb.login()` interactively prompt for your key.

---

## Section 4: Seed Configuration (Exact Reproducibility)

Exact reproducibility is guaranteed across all runs via `set_seed(42)` in `main.py`:
- Python built-in `random.seed(42)`
- NumPy `np.random.seed(42)`
- PyTorch CPU & CUDA `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`
- CUDNN deterministic mode: `torch.backends.cudnn.deterministic = True`, `benchmark = False`
- Environment variable: `PYTHONHASHSEED = '42'`

You can also pass custom seeds via the CLI: `--seed <int>` (Default: `42`).

---

## Section 5: Question-by-Question Implementation Details & Commands

### Question 1: Training Baseline (20 points)

#### (a) CIFAR-10 Data Preparation & Augmentations
- **Normalization**: Standard CIFAR-10 statistics:
  - Mean: `(0.4914, 0.4822, 0.4465)`
  - Standard Deviation: `(0.2023, 0.1994, 0.2010)`
- **Data Augmentation Transforms** (implemented in `dataset.py`):
  1. `transforms.RandomCrop(32, padding=4)`: Prevents edge-centric spatial overfitting.
  2. `transforms.RandomHorizontalFlip()`: Accounts for horizontal symmetry of object categories.
  3. `transforms.RandomRotation(degrees=5)`: Small rotation invariance.
  4. `transforms.ColorJitter(brightness=0.1, contrast=0.1)`: Photometric invariance.
  5. `Cutout(n_holes=1, length=16)`: Regularization technique masking a $16 \times 16$ square to prevent lazy feature reliance.

#### (b) MobileNet-v2 Configuration & Training Strategy
- **CIFAR-10 Architectural Adaptation**:
  - Default MobileNet-v2 downsamples $224 \times 224$ images with `stride=2` in the initial convolution. On $32 \times 32$ CIFAR-10 images, this causes premature spatial collapse.
  - **Adjustment**: The first `Conv2d` layer uses `stride=1` and `padding=1`, preserving spatial feature resolution ($32 \times 32 \to 16 \times 16 \to 8 \times 8 \to 4 \times 4$).
- **Hyperparameters**:
  - **Width Multiplier ($\alpha$)**: `1.0` (Divisible by 8 channel adjustment).
  - **Dropout**: `0.2` in classifier.
  - **Batch Normalization**: `momentum=0.1`, `eps=1e-5`.
  - **Optimizer**: Nesterov SGD (`lr=0.05`, `momentum=0.9`, `weight_decay=5e-4`, `nesterov=True`).
  - **Learning Rate Schedule**: Cosine Annealing (`T_max=100`, `eta_min=1e-4`).
  - **Batch Size**: `128`.
  - **Epochs**: `100`.

#### (c) Command to Train Baseline Model:
```bash
python main.py --epochs 100 --batch_size 128 --lr 0.05 --optimizer nesterov_sgd --seed 42
```
*Outputs: Best model saved at `./checkpoints/best_mobilenetv2.pth`, and training/validation loss and accuracy curves saved to `./data/training_curves.png`.*

---

### Question 2: Model Compression Implementation (30 points)

#### (a) Custom Quantization Methodology
- **Weights (Symmetric Signed Quantization)**:
  $$q_{\min} = -2^{b-1} + 1, \quad q_{\max} = 2^{b-1} - 1, \quad z = 0$$
  $$\text{scale} = \frac{\max(|w_{\min}|, |w_{\max}|)}{q_{\max}}, \quad q = \text{clamp}\left(\left\lfloor \frac{w}{\text{scale}} \right\rceil, q_{\min}, q_{\max}\right)$$
  - **Per-Channel Scaling**: For each output filter $c_{\text{out}}$, scale is calculated independently along $(C_{\text{in}}, K_h, K_w)$. This is critical for depthwise separable convs where individual filters have drastically distinct variance.
- **Activations (Asymmetric Unsigned Quantization)**:
  - Since post-ReLU6 activations are non-negative ($x \in [0, 6]$), unsigned quantization maximizes dynamic range:
    $$q_{\min} = 0, \quad q_{\max} = 2^b - 1$$
    $$\text{scale} = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \quad z = \text{round}\left(-\frac{x_{\min}}{\text{scale}}\right)$$
  - **Dynamic Calibration & Freeze**: Running min/max stats are tracked across training batches, then frozen before inference.

#### (b) Applying to MobileNet-v2
- **Deep Recursive Replacement**: `swap_to_quant_modules` traverses deep nested `InvertedResidual` blocks, replacing:
  - `nn.Conv2d` (including regular pointwise and depthwise `groups=in_channels`) $\to$ `QuantConv2d`.
  - `nn.Linear` $\to$ `QuantLinear`.
  - `nn.ReLU6` / `nn.ReLU` $\to$ `nn.Sequential(ReLU6, ActFakeQuant)`.

#### (c) Storage Overhead Calculation (RAM/VRAM Footprint)
$$\text{Total In-Memory Size} = \text{Weights Size} + \text{Biases Size} + \text{Metadata Overhead}$$
- **Quantized Weights**: $\sum (\text{num\_elements} \times b_{\text{weight}} / 8)$ bytes.
- **Biases (FP32)**: $\sum (\text{num\_biases} \times 4)$ bytes.
- **Weight Metadata Overhead**:
  - Per-channel: $4 \text{ bytes (FP32 scale)} \times C_{\text{out}}$ per convolution/linear layer.
- **Activation Metadata Overhead**:
  - Per-tensor: $2 \times 4 \text{ bytes (FP32 scale + zero point)}$ per activation module.

---

### Question 3: Compression Results & W&B Parallel Coordinates Sweep

To execute the required **$\ge 8$ simulation runs** and automatically log them to W&B:
```bash
python sweep_wandb.py --checkpoint ./checkpoints/best_mobilenetv2.pth --calib_batches 20
```

The script evaluates **12 configurations**:
- **W8A8, W8A6, W8A4** (Per-channel)
- **W8A8** (Per-tensor baseline comparison)
- **W6A8, W6A6, W6A4** (Per-channel)
- **W6A6** (Per-tensor)
- **W4A8, W4A6, W4A4** (Per-channel)
- **W4A4** (Per-tensor)

#### Viewing Parallel Coordinates Plot in W&B:
1. Navigate to your project on [wandb.ai](https://wandb.ai).
2. Click **Add Panel** $\to$ **Parallel Coordinates**.
3. Select dimensions: `weight_bits`, `activation_bits`, `per_channel`, `quant_test_acc`, `model_size_mb`, `total_model_compression_ratio`.
4. Export and embed into your submission PDF.

---

### Question 4: Compression Analysis

Run standalone evaluation for any bit-width setting:
```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8 --checkpoint ./checkpoints/best_mobilenetv2.pth
```
```bash
python test.py --weight_quant_bits 4 --activation_quant_bits 4 --checkpoint ./checkpoints/best_mobilenetv2.pth
```

#### Metrics Reported by `test.py`:
1. **Weight Compression Ratio**:
   $$\text{CR}_{\text{weights}} = \frac{\text{FP32 Weight Bytes}}{\text{Quantized Weight Bytes + Weight Metadata Overhead}}$$
2. **Activation Compression Ratio & Measurement Methodology**:
   - **Methodology**: In PyTorch, forward hooks (`module.register_forward_hook`) are registered on all intermediate leaf modules during an inference pass with input shape $(1, 3, 32, 32)$.
   - The total number of elements produced across all feature maps is accumulated:
     $$\text{FP32 RAM} = \text{Elements} \times 4 \text{ bytes}, \quad \text{Quantized RAM} = \text{Elements} \times \frac{b_{\text{act}}}{8} \text{ bytes}$$
     $$\text{CR}_{\text{activations}} = \frac{\text{FP32 RAM}}{\text{Quantized RAM}} \approx \frac{32}{b_{\text{act}}}$$
3. **Final Approximated In-Memory Model Size (MB)**: Reported with full overhead breakdown.

---

### Question 5: Reproducibility & Repository

- **GitHub Repository Link**: [Insert your GitHub URL here]
- **Exact Execution Environment**:
  - Python 3.10+
  - PyTorch 2.0+
  - CUDA 12.0+
  - Random Seed: `42`
