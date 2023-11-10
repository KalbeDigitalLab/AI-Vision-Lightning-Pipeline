import pytest
import torch

from src.models.components.layers.basic import ResNet18


@pytest.fixture
def ResNet18_fixture():
    """Fixture for creating a ResNet-18 model instance."""
    return ResNet18(3, 3)


def test_basic_layer(ResNet18_fixture):
    """Test the basic layer of the ResNet-18 model."""
    model = ResNet18_fixture

    input = torch.rand(1, 3, 256, 256)
    output = model(input)

    assert output.shape == torch.Size([1, 3])
