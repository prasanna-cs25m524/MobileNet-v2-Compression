import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# Core quantization math
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        xmin = torch.zeros_like(xmin)
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
    q = torch.round(x / scale + zp)
    return q.clamp(qmin, qmax)


def dequantize(q, scale, zp):
    return (q - zp) * scale


# Activation fake quantization with calibration
class ActFakeQuant(nn.Module):
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
        cur_min = x.detach().min()
        cur_max = x.detach().max()
        if cur_min < self.min_val:
            self.min_val.copy_(cur_min)
        if cur_max > self.max_val:
            self.max_val.copy_(cur_max)

    def freeze(self):
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
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)


# Quantized Conv2d (supports per-channel and per-tensor)
class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 weight_bits=8, per_channel=True):
        super(QuantConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.weight_bits = weight_bits
        self.per_channel = per_channel

        shape = (out_channels, 1, 1, 1) if per_channel else (1,)
        self.register_buffer("scale", torch.ones(shape))
        self.register_buffer("zp", torch.zeros(shape))
        self.frozen = False
        self.qmin, self.qmax = None, None

    def freeze(self):
        w = self.weight.detach()
        if self.per_channel:
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


# Quantized Linear layer
class QuantLinear(nn.Linear):
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


# Recursively swap Conv2d, Linear, ReLU/ReLU6 inside MobileNetV2
def swap_to_quant_modules(model, weight_bits=8, act_bits=8, per_channel=True,
                          activations_unsigned=True):
    def _swap_recursive(parent):
        for name, child in list(parent.named_children()):
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
                setattr(parent, name, qconv)

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
                setattr(parent, name, qlin)

            elif isinstance(child, (nn.ReLU, nn.ReLU6)):
                inplace_flag = getattr(child, "inplace", False)
                seq = nn.Sequential(OrderedDict([
                    ("act", type(child)(inplace=inplace_flag)),
                    ("aq", ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned))
                ]))
                setattr(parent, name, seq)
            else:
                _swap_recursive(child)

    _swap_recursive(model)


def freeze_all_quant(model):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ActFakeQuant)):
            mod.freeze()


# Memory calculation and overhead tracking
def compute_model_in_memory_size(model, weight_bits=8, per_channel=True, act_bits=8):
    total_weight_elements = 0
    total_bias_elements = 0
    weight_metadata_bytes = 0

    for name, p in model.named_parameters():
        if "weight" in name:
            total_weight_elements += p.numel()
            if per_channel:
                weight_metadata_bytes += p.size(0) * 4  # 4 bytes scale per output channel
            else:
                weight_metadata_bytes += 4             # 4 bytes scale per tensor
        elif "bias" in name:
            total_bias_elements += p.numel()

    act_modules_count = sum(1 for m in model.modules() if isinstance(m, ActFakeQuant))
    act_metadata_bytes = act_modules_count * 8  # 4 bytes scale + 4 bytes zero point

    fp32_weight_bytes = total_weight_elements * 4
    fp32_bias_bytes = total_bias_elements * 4
    fp32_total_bytes = fp32_weight_bytes + fp32_bias_bytes

    quant_weight_bytes = total_weight_elements * (weight_bits / 8.0)
    total_metadata_bytes = weight_metadata_bytes + act_metadata_bytes
    quant_model_bytes = quant_weight_bytes + fp32_bias_bytes + total_metadata_bytes

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
    print("=" * 60)
    print("         Model Compression & Memory Report")
    print("=" * 60)
    print(f"Config: Weights={summary['weight_bits']}-bit, Activations={summary['act_bits']}-bit")
    print(f"Per-Channel Scaling: {summary['per_channel']}")
    print("-" * 60)
    print(f"FP32 Model Size          : {summary['fp32_model_mb']:.3f} MB")
    print(f"Quantized Weights Size   : {summary['quant_weight_mb']:.3f} MB")
    print(f"Biases (FP32) Size       : {summary['bias_mb']:.4f} MB")
    print(f"Metadata Overhead        : {summary['total_overhead_kb']:.2f} KB")
    print("-" * 60)
    print(f"In-Memory Model Size     : {summary['quant_model_mb']:.3f} MB")
    print(f"Weight Compression Ratio : {summary['compression_ratio_weights']:.2f}x")
    print(f"Total Compression Ratio  : {summary['compression_ratio_model']:.2f}x")
    if act_summary is not None:
        print(f"Activation Comp. Ratio   : {act_summary['compression_ratio']:.2f}x")
        print(f"FP32 Activations (1 img) : {act_summary['fp32_kb']:.2f} KB")
        print(f"Quant Activations (1 img): {act_summary['quant_kb']:.2f} KB")
    print("=" * 60)
