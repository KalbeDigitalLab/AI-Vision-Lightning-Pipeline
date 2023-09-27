import pytest
import torch

from src.models.components.resnet18 import ResNet18


def test_resnet18():
    model = ResNet18(1, 5)
    input = torch.rand(1, 1, 256, 256)
    output = model(input)

    assert output.shape == torch.Size([1, 5])
