from typing import Optional

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet Basic Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int, optional
        Convolution stride size, by default 1
    identity_downsample : Optional[torch.nn.Module], optional
        Downsampling layer, by default None
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 identity_downsample: Optional[torch.nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward computation."""
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Apply an operation to the identity output.
        # Useful to reduce the layer size and match from conv2 output
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
