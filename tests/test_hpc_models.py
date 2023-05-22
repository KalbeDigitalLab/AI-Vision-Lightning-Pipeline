import pytest
import torch

from src.models.components import hypercomplex_models as hpc_models
from src.models.components.layers import hypercomplex_layers as hpc_layers


@pytest.mark.parametrize('batch_size,dimension,n_size,before_gap_output,gap_output,visualize', [
    [2, (2, 300, 250), 2, True, False, False],
    [2, (2, 300, 250), 2, False, True, False],
    [2, (2, 300, 250), 2, False, False, False],
    [2, (2, 300, 250), 2, False, False, True],
])
def test_phcresnet18_2views(batch_size, dimension, n_size, before_gap_output, gap_output, visualize):
    input = torch.rand([batch_size, *dimension])
    model = hpc_models.PHCResNet18(channels=2, n=n_size, num_classes=2,
                                   before_gap_output=before_gap_output, gap_output=gap_output, visualize=visualize)
    assert model.layer5 is None and model.layer6 is None
    output = model(input)

    if before_gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 512
        assert output.ndim == 4

    elif gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 512
        assert output.ndim == 2

    elif visualize:
        assert len(output) == 3
        assert output[0].shape[1] == 2

    else:
        assert len(output) == batch_size
        assert output.shape[1] == 2


@pytest.mark.parametrize('batch_size,dimension,n_size,before_gap_output,gap_output,visualize', [
    [2, (2, 300, 250), 2, True, False, False],
    [2, (2, 300, 250), 2, False, True, False],
    [2, (2, 300, 250), 2, False, False, False],
    [2, (2, 300, 250), 2, False, False, True],
    [2, (2, 300, 250), 2, False, False, False],
])
def test_phcresnet50_2views(batch_size, dimension, n_size, before_gap_output, gap_output, visualize):
    input = torch.rand([batch_size, *dimension])
    model = hpc_models.PHCResNet50(channels=2, n=n_size, num_classes=2,
                                   before_gap_output=before_gap_output, gap_output=gap_output, visualize=visualize)
    assert model.layer5 is None and model.layer6 is None
    output = model(input)

    if before_gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 1024
        assert output.ndim == 4

    elif gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 1024
        assert output.ndim == 2

    elif visualize:
        assert len(output) == 3
        assert output[0].shape[1] == 2

    else:
        assert len(output) == batch_size
        assert output.shape[1] == 2


@pytest.mark.parametrize('batch_size,dimension,n_size,before_gap_output,gap_output,visualize', [
    [2, (2, 300, 250), 2, True, False, False],
    [2, (2, 300, 250), 2, False, True, False],
    [2, (2, 300, 250), 2, False, False, False],
    [2, (2, 300, 250), 2, False, False, True],
    [2, (2, 300, 250), 2, False, False, False],
])
def test_phcresnet_residual(batch_size, dimension, n_size, before_gap_output, gap_output, visualize):
    input = torch.rand([batch_size, *dimension])
    model = hpc_models.PHCResNet(hpc_layers.ResidualBlock, [
                                 2, 2, 2, 2], channels=2, n=n_size, num_classes=2, before_gap_output=before_gap_output, gap_output=gap_output, visualize=visualize)
    assert model.layer5 is None and model.layer6 is None
    output = model(input)

    if before_gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 512
        assert output.ndim == 4

    elif gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 512
        assert output.ndim == 2

    elif visualize:
        assert len(output) == 3
        assert output[0].shape[1] == 2

    else:
        assert len(output) == batch_size
        assert output.shape[1] == 2


@pytest.mark.parametrize('batch_size,dimension,n_size,before_gap_output,gap_output,visualize', [
    [2, (2, 300, 250), 2, True, False, False],
    [2, (2, 300, 250), 2, False, True, False],
    [2, (2, 300, 250), 2, False, False, False],
    [2, (2, 300, 250), 2, False, False, True],
    [2, (2, 300, 250), 2, False, False, False],
])
def test_phcresnet_bottleneck(batch_size, dimension, n_size, before_gap_output, gap_output, visualize):
    input = torch.rand([batch_size, *dimension])
    model = hpc_models.PHCResNet(hpc_layers.Bottleneck, [
                                 3, 4, 6, 3], channels=2, n=n_size, num_classes=2, before_gap_output=before_gap_output, gap_output=gap_output, visualize=visualize)
    assert model.layer5 is None and model.layer6 is None
    output = model(input)

    if before_gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 1024
        assert output.ndim == 4

    elif gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 1024
        assert output.ndim == 2

    elif visualize:
        assert len(output) == 3
        assert output[0].shape[1] == 2

    else:
        assert len(output) == batch_size
        assert output.shape[1] == 2


@pytest.mark.parametrize('batch_size,dimension,n_size,before_gap_output,gap_output,visualize', [
    [2, (2, 300, 250), 2, True, False, False],
    [2, (2, 300, 250), 2, False, True, False],
    [2, (2, 300, 250), 2, False, False, False],
    [2, (2, 300, 250), 2, False, False, True],
    [2, (2, 300, 250), 2, False, False, False],
])
def test_phcresnet_topblock_bottleneck(batch_size, dimension, n_size, before_gap_output, gap_output, visualize):
    input = torch.rand([batch_size, *dimension])
    model = hpc_models.PHCResNet(hpc_layers.Bottleneck, [
                                 3, 4, 6, 3], channels=2, n=n_size, num_classes=2, before_gap_output=before_gap_output, gap_output=gap_output, visualize=visualize)
    model.add_top_blocks(num_classes=2)
    assert model.layer5 is not None
    assert model.layer6 is not None
    output = model(input)

    if before_gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 1024
        assert output.ndim == 4

    elif gap_output:
        assert len(output) == batch_size
        assert output.shape[1] == 1024
        assert output.ndim == 2

    elif visualize:
        assert len(output) == 3
        assert output[0].shape[1] == 2

    else:
        assert len(output) == batch_size
        assert output.shape[1] == 2


@pytest.mark.parametrize('batch_size,dimension,shared', [
    [2, (300, 250), True],
    [4, (300, 250), False],
])
def test_physbonet(batch_size, dimension, shared):
    input = tuple([torch.rand([batch_size, 2, *dimension]) for _ in range(2)])
    model = hpc_models.PHYSBOnet(n=2, shared=shared, num_classes=2)
    output = model(input)

    assert output.ndim == 2
    assert output.shape[1] == 4


@pytest.mark.parametrize('batch_size,dimension,n_size,visualize', [
    [2, (300, 250), 4, True],
    [4, (300, 250), 4, False],
    [4, (300, 250), 4, False],
])
def test_physenet(batch_size, dimension, n_size, visualize):
    input = tuple([torch.rand([batch_size, 2, *dimension]) for _ in range(2)])

    model = hpc_models.PHYSEnet(n=n_size, num_classes=2, visualize=visualize)
    output = model(input)

    if visualize:
        assert len(output) == 5
    else:
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 2
        assert output.shape[1] == 4
