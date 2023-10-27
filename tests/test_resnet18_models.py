import pytest
import torch

from src.models.components.resnet18 import ResNet18


def test_resnet18():
    """Testing ResNet-18 for Body Part Classification.

    Construct model architecture for body part classification task.

    Architecture details:
        Classification label = ['abdominal', 'adult', 'pediatric', 'spine', 'others'],
        Number of channels = 1,
        Input size = (256, 256),
    """
    model = ResNet18(1, 5)
    input = torch.rand(1, 1, 256, 256)
    output = model(input)

    assert output.shape == torch.Size([1, 5])
