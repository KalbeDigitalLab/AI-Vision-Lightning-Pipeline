from typing import Optional

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    ResNet Basic Block.

    This class defines a basic building block for ResNet architectures. It consists of two convolutional
    layers with batch normalization and a ReLU activation function. Optionally, it can include an
    identity downsample layer to match the dimensions of the input and output when the stride is not 1.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Convolution stride size, by default 1.
    identity_downsample : Optional[torch.nn.Module], optional
        Downsampling layer, by default None.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Apply forward computation.
    """

    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: int = 1,
                identity_downsample: Optional[torch.nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size = 3,
                              stride = stride,
                              padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1)
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

class ResNet18(nn.Module):
    """
    Construct ResNet-18 Model.

    This class defines the ResNet-18 architecture, including convolutional layers, basic blocks, and
    fully connected layers for classification.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    num_classes : int
        Number of class outputs.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Apply forward computation.
    """

    def __init__(self, input_channels, num_classes):

        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               64, kernel_size = 7,
                              stride = 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3,
                                   stride = 2,
                                   padding = 1)

        self.layer1 = self._make_layer(64, 64, stride = 1)
        self.layer2 = self._make_layer(64, 128, stride = 2)
        self.layer3 = self._make_layer(128, 256, stride = 2)
        self.layer4 = self._make_layer(256, 512, stride = 2)

        # Last layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def identity_downsample(self, in_channels: int, out_channels: int) -> nn.Module:
        """Downsampling block to reduce the feature sizes."""
        return nn.Sequential(
             nn.Conv2d(in_channels,
                       out_channels,
                       kernel_size = 3,
                       stride = 2,
                       padding = 1),
            nn.BatchNorm2d(out_channels)
        )

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Create sequential basic block."""
        identity_downsample = None

        # Add downsampling function
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
                    BasicBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
                    BasicBlock(out_channels, out_channels)
                    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x