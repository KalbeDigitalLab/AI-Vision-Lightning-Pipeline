import pytest
import torch

import src.models.components.layers.hypercomplex_ops as hpc_ops


def test_catch_invalid_check_input():
    input = torch.rand(10, 100, 25, 25, 5, 3)
    with pytest.raises(Exception) as exec_info:
        hpc_ops.check_input(input)
        assert exec_info == f'Quaternion linear accepts only input of dimension 2 or 3. Quaternion conv accepts up to 5 dim input.dim = {str(input.dim())}'

    input = torch.rand(10, 75, 3)
    with pytest.raises(Exception) as exec_info:
        hpc_ops.check_input(input)
        assert exec_info == f'Quaternion Tensors must be divisible by 4. input.size()[1] = {75}'


def test_run_valid_check_input():
    input = torch.rand(10, 25, 100)
    assert hpc_ops.check_input(input) == None

    input = torch.rand(10, 100, 25, 25)
    assert hpc_ops.check_input(input) == None


@pytest.mark.parametrize('input', [
    torch.rand(10, 100),
    torch.rand(10, 25, 100),
    torch.rand(10, 100, 25, 25),
    torch.rand(10, 100, 25, 25, 25),
])
def test_run_get_r(input):
    if input.dim() == 2:
        return input.narrow(1, 0, 25)
    if input.dim() == 3:
        return input.narrow(2, 0, 25)
    if input.dim() >= 4:
        return input.narrow(1, 0, 25)
    output = hpc_ops.get_r(input)
    assert torch.allclose(input, output)


@pytest.mark.parametrize('input', [
    torch.rand(10, 100),
    torch.rand(10, 25, 100),
    torch.rand(10, 100, 25, 25),
    torch.rand(10, 100, 25, 25, 25),
])
def test_run_get_i(input):
    if input.dim() == 2:
        return input.narrow(1, 25, 25)
    if input.dim() == 3:
        return input.narrow(2, 25, 25)
    if input.dim() >= 4:
        return input.narrow(1, 25, 25)
    output = hpc_ops.get_i(input)
    assert torch.allclose(input, output)


@pytest.mark.parametrize('input', [
    torch.rand(10, 100),
    torch.rand(10, 25, 100),
    torch.rand(10, 100, 25, 25),
    torch.rand(10, 100, 25, 25, 25),
])
def test_run_get_j(input):
    if input.dim() == 2:
        return input.narrow(1, 50, 25)
    if input.dim() == 3:
        return input.narrow(2, 50, 25)
    if input.dim() >= 4:
        return input.narrow(1, 50, 25)
    output = hpc_ops.get_j(input)
    assert torch.allclose(input, output)


@pytest.mark.parametrize('input', [
    torch.rand(10, 100),
    torch.rand(10, 25, 100),
    torch.rand(10, 100, 25, 25),
    torch.rand(10, 100, 25, 25, 25),
])
def test_run_get_k(input):
    if input.dim() == 2:
        return input.narrow(1, 75, 25)
    if input.dim() == 3:
        return input.narrow(2, 75, 25)
    if input.dim() >= 4:
        return input.narrow(1, 75, 25)
    output = hpc_ops.get_k(input)
    assert torch.allclose(input, output)
