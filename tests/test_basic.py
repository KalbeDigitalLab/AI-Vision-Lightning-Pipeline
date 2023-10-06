import pytest
import torch
from src.models.components.layers.basic import ResNet18

@pytest.fixture
def ResNet18_fixture():
    return ResNet18(3,3) 

def test_basic_layer(ResNet18_fixture):
    model = ResNet18_fixture

    input = torch.rand(1, 3, 256, 256)
    output = model(input)

    assert output.shape == torch.Size([1, 3])