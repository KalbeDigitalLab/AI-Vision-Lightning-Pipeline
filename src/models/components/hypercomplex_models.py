from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.layers import hypercomplex_layers as hpc_layer
from src.models.components.utils import utils as m_utils


class ResidualBlock(nn.Module):
    """Residual PH Convolution Block.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L15
    Reference:
    - arxiv.org/abs/1512.03385
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    in_planes : int
        Input feature size
    planes : int
        Output feature size
    stride : int, optional
        Convolution stride size, by default 1
    n : int, optional
        Number of dimensions, by default 4
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, n: int = 4):
        super().__init__()
        self.conv1 = hpc_layer.PHConv(n,
                                      in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = hpc_layer.PHConv(n, planes, planes, kernel_size=3,
                                      stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                hpc_layer.PHConv(n, in_planes, self.expansion*planes,
                                 kernel_size=1, stride=stride,),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck PH Convolution Block.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L43
    Reference:
    - arxiv.org/abs/1512.03385
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    in_planes : int
        Input feature size
    planes : int
        Output feature size
    stride : int, optional
        Convolution stride size, by default 1
    n : int, optional
        Number of dimensions, by default 4
    """
    expansion = 2

    def __init__(self, in_planes: int, planes: int, stride: int = 1, n: int = 4):
        super().__init__()
        self.conv1 = hpc_layer.PHConv(n, in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = hpc_layer.PHConv(n, planes, planes, kernel_size=3,
                                      stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = hpc_layer.PHConv(n, planes, self.expansion * planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                hpc_layer.PHConv(n, in_planes, self.expansion*planes,
                                 kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """Encoder Block for Feature Extraction.

    This block has been designed to produce 128 feature planes.
    This block is part of PHYSBONet.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L150
    Reference:
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    channels : int
        Input feature size
    n : int
        Number of dimensions
    """

    def __init__(self, channels: int, n: int):
        super().__init__()
        self.in_planes = 64

        self.conv1 = hpc_layer.PHConv(n, channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1, n=n)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2, n=n)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int, n: int):
        """Create sequential layer of Residual Blocks."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class SharedBottleneck(nn.Module):
    """SharedBottleneck Block for feature fusion.

    This block has been designed to produce 512 feature planes.
    This block is part of PHYSBONet.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L178
    Reference:
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    channels : int
        Input feature size
    n : int
        Number of dimensions
    """

    def __init__(self, n, in_planes):
        super().__init__()
        self.in_planes = in_planes

        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2, n=n)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2, n=n)
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int, n: int):
        """Create sequential layer of Residual and Bottleneck Blocks."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        n, c, _, _ = out.size()
        out = out.view(n, c, -1).mean(-1)
        return out


class Classifier(nn.Module):
    """Refined Classifier Block.

    This block will use a deeper / refined Bottleneck layers
    before parsing it to a Linear Layer.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L209
    Reference:
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    n : int
        Number of dimensions
    num_classes : int
        Number of output classes
    in_planes : int, optional
        Input feature size, by default 512
    visualize : bool, optional
        Return the activation maps before linear layer, by default False
    """

    def __init__(self, n: int, num_classes: int, in_planes=512, visualize=False):
        super().__init__()
        self.in_planes = in_planes
        self.visualize = visualize

        # Refiner blocks
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int, n: int):
        """Create sequential layer of Bottleneck Blocks."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        out = self.layer5(x)
        feature_maps = self.layer6(out)

        n, c, _, _ = feature_maps.size()
        out = feature_maps.view(n, c, -1).mean(-1)
        out = self.linear(out)

        if self.visualize:
            return out, feature_maps

        return out


class PHCResNet(nn.Module):
    """Parameterized Hypercomplex Residual Network.

    The proposed multi-view architecture in the case of two views.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L73
    Reference:
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    block : Callable
        Type of convolution block to use.
    num_blocks : List[int, int, int, int]
        Number of repetition in each blocks
    channels : int, optional
        Number of input channels, by default 4
    n : int, optional
        Number of dimensions, by default 4
    num_classes : int, optional
        Number of output classes, by default 10
    before_gap_output : bool, optional
        Return the output before refiner blocks and gap, by default False
    gap_output : bool, optional
        Return the output after gap and before final linear layer, by default False
    visualize : bool, optional
        Return the final output and activation maps at two different levels, by default False
    """

    def __init__(self, block: Callable, num_blocks: List[int, int, int, int], channels=4, n=4, num_classes=10, before_gap_output=False, gap_output=False, visualize=False):
        super().__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.n = n
        self.before_gap_out = before_gap_output
        self.gap_output = gap_output
        self.visualize = visualize

        self.conv1 = hpc_layer.PHConv(n, channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n=n)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n=n)

        # Refiner blocks
        self.layer5 = None
        self.layer6 = None

        if not before_gap_output and not gap_output:
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def add_top_blocks(self, num_classes=1):
        """Add refiner blocks before final linear layer."""
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2, n=self.n)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2, n=self.n)

        if not self.before_gap_out and not self.gap_output:
            self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        """Create sequential layer of Convolution Blocks."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple]:
        """Compute the forward computation."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out4 = self.layer4(out)

        if self.before_gap_out:
            return out4

        if self.layer5:
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)

        # global average pooling (GAP)
        n, c, _, _ = out6.size()
        out = out6.view(n, c, -1).mean(-1)

        if self.gap_output:
            return out

        out = self.linear(out)

        if self.visualize:
            # return the final output and activation maps at two different levels
            return out, out4, out6
        return out


class PHYSBOnet(nn.Module):
    """Parameterized Hypercomplex Shared Bottleneck Network.

    It's based on an initial breast-level focus and a consequent
    patient-level one, where each encoder takes as input two views
    (CC and MLO) and has the objective of learning a latent
    representation of the ipsilateral views. The learned latent
    representations are then merged together and processed by the bottleneck.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L245
    Reference:
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    n : int
        Number of dimensions
    shared : bool, optional
        Share the Bottleneck between the two sides, by default True
    num_classes : int, optional
        Number of output classes, by default 1
    weights : str, optional
        path to pretrained weights of patch classifier for Encoder branches, by default None
    """

    def __init__(self, n: int, shared: bool = True, num_classes: int = 1, weights: Optional[str] = None):
        super().__init__()

        self.shared = shared

        self.encoder_sx = Encoder(channels=2, n=2)
        self.encoder_dx = Encoder(channels=2, n=2)

        self.shared_resnet = SharedBottleneck(n, in_planes=128 if shared else 256)

        if weights:
            m_utils.load_weights(self.encoder_sx, weights)
            m_utils.load_weights(self.encoder_dx, weights)

        self.classifier_sx = nn.Linear(1024, num_classes)
        self.classifier_dx = nn.Linear(1024, num_classes)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute the forward computation."""
        x_sx, x_dx = x

        # Apply Encoder
        out_sx = self.encoder_sx(x_sx)
        out_dx = self.encoder_dx(x_dx)

        # Shared layers
        if self.shared:
            out_sx = self.shared_resnet(out_sx)
            out_dx = self.shared_resnet(out_dx)

            out_sx = self.classifier_sx(out_sx)
            out_dx = self.classifier_dx(out_dx)

        else:  # Concat version
            out = torch.cat([out_sx, out_dx], dim=1)
            out = self.shared_resnet(out)
            out_sx = self.classifier_sx(out)
            out_dx = self.classifier_dx(out)

        out = torch.cat([out_sx, out_dx], dim=0)
        return out


class PHYSEnet(nn.Module):
    """Parameterized Hypercomplex Shared Encoder Network.

    It has a broader focus on the patient-level analysis through,
    which takes as input two ipsilateral views and whose weights
    between the two sides (left and right breast) are shared to
    jointly analyze the whole information of the patient.

    Source: https://github.com/ispamm/PHBreast/blob/main/models/phc_models.py#L295
    Reference:
    - arxiv.org/abs/2204.05798

    Parameters
    ----------
    n : int, optional
        Number of dimensions, by default 2
    num_classes : int, optional
        Number of output classes, by default 1
    weights : _type_, optional
        Path to pretrained weights of patch classifier for PHCResNet18 encoder or path to whole-image classifier, by default None
    patch_weights : bool, optional
        True if the weights correspond to patch classifier, False if they are whole-image, by default True
    visualize : bool, optional
        Return the encoder and refined block activation maps, by default False
    """

    def __init__(self, n=2, num_classes=1, weights=None, patch_weights=True, visualize=False):
        super().__init__()
        self.visualize = visualize
        self.phcresnet18 = PHCResNet18(n=2, num_classes=num_classes, channels=2, before_gap_output=True)

        if weights:
            print('Loading weights for phcresnet18 from ', weights)
            m_utils.load_weights(self.phcresnet18, weights)

        self.classifier_sx = Classifier(n, num_classes, visualize=visualize)
        self.classifier_dx = Classifier(n, num_classes, visualize=visualize)

        if not patch_weights and weights:
            print('Loading weights for classifiers from ', weights)
            m_utils.load_weights(self.classifier_sx, weights)
            m_utils.load_weights(self.classifier_dx, weights)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        """Compute the forward computation."""
        x_sx, x_dx = x

        # Apply Encoder
        out_enc_sx = self.phcresnet18(x_sx)
        out_enc_dx = self.phcresnet18(x_dx)

        if self.visualize:
            out_sx, act_sx = self.classifier_sx(out_enc_sx)
            out_dx, act_dx = self.classifier_dx(out_enc_dx)
        else:
            # Apply refiner blocks + classifier
            out_sx = self.classifier_sx(out_enc_sx)
            out_dx = self.classifier_dx(out_enc_dx)

        out = torch.cat([out_sx, out_dx], dim=0)

        if self.visualize:
            return out, out_enc_sx, out_enc_dx, act_sx, act_dx

        return out


def PHCResNet18(channels=4, n=4, num_classes=10, before_gap_output=False, gap_output=False, visualize=False):
    """Default constructor for PHCResnet-18."""
    return PHCResNet(ResidualBlock,
                     [2, 2, 2, 2],
                     channels=channels,
                     n=n,
                     num_classes=num_classes,
                     before_gap_output=before_gap_output,
                     gap_output=gap_output,
                     visualize=visualize)


def PHCResNet50(channels=4, n=4, num_classes=10):
    """Default constructor for PHCResnet-50."""
    return PHCResNet(Bottleneck, [3, 4, 6, 3], channels=channels, n=n, num_classes=num_classes)
