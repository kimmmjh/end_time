from torch import nn


import torch
import torch.nn.functional as F

class SemiCircularConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=0)
        self.pad_size = kernel_size // 2

    def forward(self, x):
        # x is (B, C, D, H, W) -> (batch, channels, time, x, y)
        # Pad W (Y) and H (X) circularly for Toric Code boundaries
        x = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size, 0, 0), mode='circular')
        # Pad D (Time) with zeros, because Time does not loop!
        x = F.pad(x, (0, 0, 0, 0, self.pad_size, self.pad_size), mode='constant', value=0)
        return self.conv(x)

def circular_conv_3d(in_channels: int, out_channels: int, kernel_size: int, bias=False) -> nn.Module:
    """
    Return a semi-circular padded 3d convolution layer.
    """
    return SemiCircularConv3d(in_channels, out_channels, kernel_size, bias=bias)
