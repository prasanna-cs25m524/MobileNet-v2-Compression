"""
MobileNetV2 Architecture adapted for CIFAR-10.
Includes support for configurable width multiplier (alpha), dropout, and batch norm settings.
Adapted to prevent excessive spatial downsampling on 32x32 CIFAR-10 images.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    """
    Ensures that all layer channel counts are divisible by 8 (standard for hardware efficiency).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """
    Inverted Residual block:
      1. 1x1 Conv (Pointwise) -> expansion (if expand_ratio != 1)
      2. 3x3 Depthwise Conv (groups = in_channels)
      3. 1x1 Conv (Pointwise) -> projection back (Linear, no non-linearity)
      4. Residual skip-connection if stride == 1 and in_channels == out_channels
    """
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

        # 3x3 depthwise convolution
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # 1x1 linear projection (without activation)
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2, norm_layer=None):
        super(MobileNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        # Inverted residual settings for CIFAR-10:
        # [t: expansion factor, c: output channels, n: repeat count, s: stride]
        # We use stride=1 on the first inverted residual stage to avoid premature downsampling of 32x32 images
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # stride 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer: Conv2d with stride=1 (CIFAR-10 adaptation)
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

        features = [ConvBNReLU(3, input_channel, kernel_size=3, stride=1, norm_layer=norm_layer)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        # Building last pointwise convolution layer
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)

        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes=10, width_mult=1.0, dropout=0.2):
    """Factory function to instantiate MobileNetV2 with user configuration."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, dropout=dropout)
