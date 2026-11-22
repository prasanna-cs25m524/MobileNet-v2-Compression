"""
quantize.py: Custom Quantization Engine for MobileNet-v2 (CIFAR-10)

Key Features:
1. Pure PyTorch implementation without external compression APIs (torch.ao, etc.).
2. Configurable bit-widths (e.g., 8-bit, 6-bit, 4-bit) for both weights and activations.
3. Per-channel (per-filter) & Per-tensor symmetric quantization for weights (crucial for depthwise convs).
4. Asymmetric unsigned quantization for post-ReLU6 activations with dynamic calibration & freeze.
5. Recursive module swapping supporting deep InvertedResidual blocks in MobileNet-v2.
6. Explicit in-memory RAM/VRAM footprint and metadata storage overhead calculations.
"""

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Core Uniform Quantization Helpers
# ==============================================================================

def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    """
    Returns (scale, zero_point, qmin, qmax) for uniform quantization.
    - unsigned=True  -> [0, 2^b - 1] (Used for non-negative post-ReLU/ReLU6 activations)
    - unsigned=False -> [-2^(b-1)+1, 2^(b-1)-1] (Symmetric integer range around 0 for weights)
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        xmin = torch.zeros_like(xmin)  # Enforce non-negative zero for tighter activation range
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, int(qmin), int(qmax)


def quantize(x, scale, zp, qmin, qmax):
    """Uniform quantization with rounding and clamping."""
    q = torch.round(x / scale + zp)
    return q.clamp(qmin, qmax)


def dequantize(q, scale, zp):
    """Dequantization back to floating point."""
    return (q - zp) * scale


# ==============================================================================
# 2. Activation Fake-Quantization Module (Post-ReLU / Post-ReLU6)
# ==============================================================================

class ActFakeQuant(nn.Module):
    """
    Activation fake-quantization with configurable bit-width.
    Calibrates running min/max statistics across calibration batches, then freezes.
    Placed after ReLU / ReLU6 activations (unsigned=True).
    """
    def __init__(self, n_bits=8, unsigned=True):
        super(ActFakeQuant, self).__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin, self.qmax = None, None

    def update_stats(self, x):
        """Update calibration min/max ranges."""
        cur_min = x.detach().min()
        cur_max = x.detach().max()
        if cur_min < self.min_val:
            self.min_val.copy_(cur_min)
        if cur_max > self.max_val:
            self.max_val.copy_(cur_max)

    def freeze(self):
        """Compute final scale and zero-point and freeze calibration."""
        if not self.frozen:
            s, zp, qmin, qmax = qparams_from_minmax(
                self.min_val, self.max_val, n_bits=self.n_bits, unsigned=self.unsigned
            )
            self.scale.copy_(s)
            self.zp.copy_(zp)
            self.qmin = qmin
            self.qmax = qmax
            self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.update_stats(x)
            return x
        # Quantize and dequantize (simulating integer precision arithmetic)
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)


# ==============================================================================
# 3. Quantized Layers: QuantConv2d and QuantLinear
# ==============================================================================

class QuantConv2d(nn.Conv2d):
    """
    Convolution layer with custom weight quantization.
    Supports both per-channel (essential for depthwise separable convs) and per-tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 weight_bits=8, per_channel=True):
        super(QuantConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.weight_bits = weight_bits
        self.per_channel = per_channel

        # Buffers for quantization parameters
        shape = (out_channels, 1, 1, 1) if per_channel else (1,)
        self.register_buffer("scale", torch.ones(shape))
        self.register_buffer("zp", torch.zeros(shape))
        self.frozen = False
        self.qmin, self.qmax = None, None

    def freeze(self):
        """Computes quantization parameters for static weights."""
        w = self.weight.detach()
        if self.per_channel:
            # Min and max along each output filter: (out_channels, 1, 1, 1)
            w_reshaped = w.view(w.size(0), -1)
            w_min = w_reshaped.min(dim=1)[0].view(w.size(0), 1, 1, 1)
            w_max = w_reshaped.max(dim=1)[0].view(w.size(0), 1, 1, 1)
        else:
            w_min = w.min()
            w_max = w.max()

        s, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.scale.copy_(s)
        self.zp.copy_(zp)
        self.qmin = qmin
        self.qmax = qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return super(QuantConv2d, self).forward(x)
        qw = quantize(self.weight, self.scale, self.zp, self.qmin, self.qmax)
        w_dequant = dequantize(qw, self.scale, self.zp)
        return F.conv2d(x, w_dequant, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    """
    Linear (fully-connected) layer with custom weight quantization.
    Supports per-channel (per-output-neuron) and per-tensor symmetric quantization.
    """
    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=8, per_channel=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.per_channel = per_channel

        shape = (out_features, 1) if per_channel else (1,)
        self.register_buffer("scale", torch.ones(shape))
        self.register_buffer("zp", torch.zeros(shape))
        self.frozen = False
        self.qmin, self.qmax = None, None

    def freeze(self):
        w = self.weight.detach()
        if self.per_channel:
            w_min = w.min(dim=1)[0].view(w.size(0), 1)
            w_max = w.max(dim=1)[0].view(w.size(0), 1)
        else:
            w_min = w.min()
            w_max = w.max()

        s, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.scale.copy_(s)
        self.zp.copy_(zp)
        self.qmin = qmin
        self.qmax = qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return super(QuantLinear, self).forward(x)
        qw = quantize(self.weight, self.scale, self.zp, self.qmin, self.qmax)
        w_dequant = dequantize(qw, self.scale, self.zp)
        return F.linear(x, w_dequant, self.bias)


# ==============================================================================
# 4. Recursive Module Replacement for Deeply Nested MobileNet-v2
# ==============================================================================

def swap_to_quant_modules(model, weight_bits=8, act_bits=8, per_channel=True,
                          activations_unsigned=True, keep_first_last_fp32=False):
    """
    Recursively replaces all Conv2d, Linear, ReLU, and ReLU6 layers in MobileNet-v2
    with QuantConv2d, QuantLinear, and ActFakeQuant.
    Handles deep nested InvertedResidual blocks properly.
    """
    def _swap_recursive(parent_module):
        for name, child in list(parent_module.named_children()):
            if isinstance(child, nn.Conv2d) and not isinstance(child, QuantConv2d):
                qconv = QuantConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    weight_bits=weight_bits,
                    per_channel=per_channel
                )
                qconv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qconv.bias.data.copy_(child.bias.data)
                setattr(parent_module, name, qconv)

            elif isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
                qlin = QuantLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    weight_bits=weight_bits,
                    per_channel=per_channel
                )
                qlin.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qlin.bias.data.copy_(child.bias.data)
                setattr(parent_module, name, qlin)

            elif isinstance(child, (nn.ReLU, nn.ReLU6)):
                inplace_flag = getattr(child, "inplace", False)
                seq = nn.Sequential(OrderedDict([
                    ("act", type(child)(inplace=inplace_flag)),
                    ("aq", ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned))
                ]))
                setattr(parent_module, name, seq)
            else:
                _swap_recursive(child)

    _swap_recursive(model)


def freeze_all_quant(model):
    """
    Finalize quantization parameters for all QuantConv2d, QuantLinear, and ActFakeQuant layers.
    Must be called after calibration batches.
    """
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ActFakeQuant)):
            mod.freeze()


# ==============================================================================
# 5. In-Memory Footprint & Metadata Storage Overhead Analysis
# ==============================================================================

def compute_model_in_memory_size(model, weight_bits=8, per_channel=True, act_bits=8):
    """
    Computes exact in-memory (RAM/VRAM) size accounting for:
      1. Baseline FP32 model size (all params as 4 bytes).
      2. Quantized weights size (num_weight_elements * weight_bits / 8 bytes).
      3. Biases (kept in FP32 = 4 bytes per bias element).
      4. Metadata storage overheads:
         - Per-channel weight scale: 4 bytes (FP32) per output channel/neuron.
         - Per-tensor activation scale + zero-point: 2 * 4 bytes per activation layer.
    Returns a detailed summary dictionary.
    """
    total_weight_elements = 0
    total_bias_elements = 0
    num_weight_tensors = 0
    weight_metadata_bytes = 0

    for name, p in model.named_parameters():
        if "weight" in name:
            total_weight_elements += p.numel()
            num_weight_tensors += 1
            if per_channel:
                weight_metadata_bytes += p.size(0) * 4  # FP32 scale per output channel
            else:
                weight_metadata_bytes += 4             # FP32 scale per tensor
        elif "bias" in name:
            total_bias_elements += p.numel()

    # Activation metadata: scale and zero-point (2 x 4 bytes per ActFakeQuant module)
    act_modules_count = sum(1 for m in model.modules() if isinstance(m, ActFakeQuant))
    act_metadata_bytes = act_modules_count * 8

    # Byte calculations
    fp32_weight_bytes = total_weight_elements * 4
    fp32_bias_bytes = total_bias_elements * 4
    fp32_total_bytes = fp32_weight_bytes + fp32_bias_bytes

    quant_weight_bytes = total_weight_elements * (weight_bits / 8.0)
    total_metadata_bytes = weight_metadata_bytes + act_metadata_bytes
    quant_model_bytes = quant_weight_bytes + fp32_bias_bytes + total_metadata_bytes

    # Compression ratios
    cr_weights = fp32_weight_bytes / max((quant_weight_bytes + weight_metadata_bytes), 1.0)
    cr_model = fp32_total_bytes / max(quant_model_bytes, 1.0)

    return {
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "per_channel": per_channel,
        "total_weight_elements": total_weight_elements,
        "total_bias_elements": total_bias_elements,
        "fp32_model_mb": fp32_total_bytes / (1024.0 * 1024.0),
        "fp32_weight_mb": fp32_weight_bytes / (1024.0 * 1024.0),
        "quant_weight_mb": quant_weight_bytes / (1024.0 * 1024.0),
        "bias_mb": fp32_bias_bytes / (1024.0 * 1024.0),
        "weight_overhead_kb": weight_metadata_bytes / 1024.0,
        "act_overhead_kb": act_metadata_bytes / 1024.0,
        "total_overhead_kb": total_metadata_bytes / 1024.0,
        "quant_model_mb": quant_model_bytes / (1024.0 * 1024.0),
        "compression_ratio_weights": cr_weights,
        "compression_ratio_model": cr_model,
    }


def print_compression_report(summary, act_summary=None):
    """Prints a clear and formatted in-memory compression report."""
    print("=" * 65)
    print("        IN-MEMORY MODEL COMPRESSION & OVERHEAD REPORT         ")
    print("=" * 65)
    print(f"Configuration: Weights={summary['weight_bits']}-bit, Activations={summary['act_bits']}-bit")
    print(f"Per-Channel Weight Scaling: {summary['per_channel']}")
    print("-" * 65)
    print(f"FP32 Baseline Model Size : {summary['fp32_model_mb']:.3f} MB")
    print(f"FP32 Weights Size        : {summary['fp32_weight_mb']:.3f} MB")
    print(f"Quantized Weights Size   : {summary['quant_weight_mb']:.3f} MB")
    print(f"Biases (FP32) Size       : {summary['bias_mb']:.4f} MB")
    print(f"Weight Metadata Overhead : {summary['weight_overhead_kb']:.2f} KB (FP32 scale factors)")
    print(f"Activation Metadata      : {summary['act_overhead_kb']:.2f} KB (scale + zero-points)")
    print(f"Total Metadata Overhead  : {summary['total_overhead_kb']:.2f} KB")
    print("-" * 65)
    print(f"Final In-Memory Model Size: {summary['quant_model_mb']:.3f} MB")
    print(f"Weight Compression Ratio : {summary['compression_ratio_weights']:.2f}x")
    print(f"Total Model Comp. Ratio  : {summary['compression_ratio_model']:.2f}x")
    if act_summary is not None:
        print(f"Activation Comp. Ratio   : {act_summary['compression_ratio']:.2f}x (Peak RAM Forward Pass)")
        print(f"FP32 Activations (1-img) : {act_summary['fp32_kb']:.2f} KB")
        print(f"Quant Activations (1-img): {act_summary['quant_kb']:.2f} KB")
    print("=" * 65)
