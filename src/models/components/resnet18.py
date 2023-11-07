import torch
from torch import nn

from src.models.components.layers.resnet_block import BasicBlock


class ResNet18(nn.Module):
    """Construct ResNet-18 Model.

    Parameters
    ----------
    input_channels : int
        Number of input channels
    num_classes : int
        Number of class outputs
    """

    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               64, kernel_size=7,
                               stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Last layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def identity_downsample(self, in_channels: int, out_channels: int) -> nn.Module:
        """Downsampling block to reduce the feature sizes."""
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Create sequential basic block."""
        identity_downsample = None

        # Add downsampling function
        if stride != 1:
            identity_downsample = self.identity_downsample(
                in_channels, out_channels)

        return nn.Sequential(
            BasicBlock(in_channels, out_channels,
                       identity_downsample=identity_downsample, stride=stride),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass method on ResNet-18 architecture.

        Reference:
            https://arxiv.org/abs/1512.03385v1
        """
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
