# This layers are borrowed from: https://github.com/eleGAN23/HyperNets
# by Eleonora Grassucci,
# Please check the original reposiotry for further explanations.

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState
from torch.nn import Module, init
from torch.nn.parameter import Parameter

from src.models.components.layers import hypercomplex_initializer as hpc_init
from src.models.components.layers import hypercomplex_ops as hpc_ops


class PHMLinear(nn.Module):
    """Parametrized Hypercomplex Multiplication Linear Layer.

    Source: https://github.com/eleGAN23/HyperNets/blob/main/layers/ph_layers.py
    Reference: arxiv.org/pdf/2102.08597

    Parameters
    ----------
    n : int
        Number of dimensions
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    cuda: bool
        Use cuda to optimize computation
    """

    def __init__(self, n: int, in_features: int, out_features: int, cuda: bool = True):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

        self.S = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, self.out_features//n, self.in_features//n))))

        self.weight = torch.zeros((self.out_features, self.in_features))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the kronecker products between 2 tensors."""
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def kronecker_product2(self) -> torch.Tensor:
        """Compute the kronecker products between 2 tensors.

        # ! Slow computation
        """
        H = torch.zeros((self.out_features, self.in_features))
        if self.cuda:
            H = H.cuda()
        for i in range(self.n):
            H = H + torch.kron(self.A[i], self.S[i])
        return H

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
        return F.linear(input, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        """Get extra str format of layer parameters."""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        """Reset the parameters using kaiming uniform."""
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.S, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class PHConv(Module):
    """Parametrized Hypercomplex Convolution Layer.

    Parameters
    ----------
    n : int
        Number of dimensions
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    kernel_size : int
        Convolution kernel size
    padding : int, optional
        Convolution padding size, by default 0
    stride : int, optional
        Convolution stride size, by default 1
    cuda: bool
        Use cuda to optimize computation
    """

    def __init__(self, n, in_features: int, out_features: int, kernel_size: int, padding: int = 0, stride: int = 1, cuda: bool = True):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.padding = padding
        self.stride = stride
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size, kernel_size))))
        self.weight = torch.zeros((self.out_features, self.in_features))
        self.kernel_size = kernel_size

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Compute the kronecker products between 2 tensors."""
        siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
        siz2 = torch.Size(torch.tensor(F.shape[-2:]))
        res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
        siz0 = res.shape[:1]
        out = res.reshape(siz0 + siz1 + siz2)
        return out

    def kronecker_product2(self) -> torch.Tensor:
        """Compute the kronecker products between 2 tensors.

        # ! Slow computation
        """
        H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
        if self.cuda:
            H = H.cuda()
        for i in range(self.n):
            kron_prod = torch.kron(self.A[i], self.F[i]).view(
                self.out_features, self.in_features, self.kernel_size, self.kernel_size)
            H = H + kron_prod
        return H

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)

        input = input.type(dtype=self.weight.type())

        return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Get extra str format of layer parameters."""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        """Reset the parameters using kaiming uniform."""
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.F, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class KroneckerConv(Module):
    """Kronecker Convolution Layer.

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    kernel_size : int
        Kernel size for covolution.
    stride : int
        Stride size for convolution
    dilatation : int, optional
        Dilation size for convolution, by default 1
    padding : int, optional
        Padding size for convolution, by default 0
    groups : int, optional
        Groups size for convolution, by default 1
    bias : bool, optional
        Enable bias computation, by default True
    init_criterion : str, optional
        Weight initialization type, by default 'he'
    weight_init : str, optional
        Weight initialization function type, by default 'quaternion'
    seed : int, optional
        Seed number for Random generator, by default None
    operation : str, optional
        Type of covolution operations, by default 'convolution2d'
    rotation : bool, optional
        Enable quaternion rotation, by default False
    quaternion_format : bool, optional
        Return in quaternion format, by default True
    scale : bool, optional
        Enable quaternoin scaling, by default False
    learn_A : bool, optional
        _description_, by default False
    cuda: bool, optional
        Use cuda to optimize computation
    first_layer : bool, optional
        Flag to build the Hamilton product on first layer, by default False
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=True, scale=False, learn_A=False, cuda=True, first_layer=False):
        super().__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': hpc_init.quaternion_init,
                      'unitary': hpc_init.unitary_init,
                      'random': hpc_init.random_init}[self.weight_init]
        self.scale = scale
        self.learn_A = learn_A
        self.cuda = cuda
        self.first_layer = first_layer

        (self.kernel_size, self.w_shape) = hpc_init.get_kernel_and_weight_shape(self.operation,
                                                                                self.in_channels, self.out_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters using custom initialization."""
        hpc_init.affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                  self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        if self.rotation:
            # return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
            #     self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
            #     self.quaternion_format, self.scale_param)
            pass
        else:
            return hpc_ops.kronecker_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                          self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation, self.learn_A, self.cuda, self.first_layer)

    def __repr__(self) -> str:
        """Get str format of layer parameters."""
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', bias=' + str(self.bias is not None) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) \
            + ', rotation=' + str(self.rotation) \
            + ', q_format=' + str(self.quaternion_format) \
            + ', operation=' + str(self.operation) + ')'


class QuaternionTransposeConv(Module):
    """Quaternion Transposed Convolution Layer..

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    kernel_size : int
        Kernel size for covolution.
    stride : int
        Stride size for convolution
    dilatation : int, optional
        Dilation size for convolution, by default 1
    padding : int, optional
        Padding size for convolution, by default 0
    output_padding : int, optional
        Addition padding to output, by default 0
    groups : int, optional
        Groups size for convolution, by default 1
    bias : bool, optional
        Enable bias computation, by default True
    init_criterion : str, optional
        Weight initialization type, by default 'he'
    weight_init : str, optional
        Weight initialization function type, by default 'quaternion'
    seed : int, optional
        Seed number for Random generator, by default None
    operation : str, optional
        Type of covolution operations, by default 'convolution2d'
    rotation : bool, optional
        Enable quaternion rotation, by default False
    quaternion_format : bool, optional
        Return in quaternion format, by default True
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):
        super().__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': hpc_init.quaternion_init,
                      'unitary': hpc_init.unitary_init,
                      'random': hpc_init.random_init}[self.weight_init]

        (self.kernel_size, self.w_shape) = hpc_init.get_kernel_and_weight_shape(self.operation,
                                                                                self.out_channels, self.in_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters using custom initialization."""
        hpc_init.affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                  self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        if self.rotation:
            return hpc_ops.quaternion_transpose_conv_rotation(input, self.r_weight, self.i_weight,
                                                              self.j_weight, self.k_weight, self.bias, self.stride, self.padding,
                                                              self.output_padding, self.groups, self.dilatation, self.quaternion_format)
        else:
            return hpc_ops.quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                                     self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                                                     self.groups, self.dilatation)

    def __repr__(self) -> str:
        """Get str format of layer parameters."""
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', bias=' + str(self.bias is not None) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', dilation=' + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) \
            + ', operation=' + str(self.operation) + ')'


class QuaternionConv(Module):
    """Quaternion Convolution Layer..

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    kernel_size : int
        Kernel size for covolution.
    stride : int
        Stride size for convolution
    dilatation : int, optional
        Dilation size for convolution, by default 1
    padding : int, optional
        Padding size for convolution, by default 0
    groups : int, optional
        Groups size for convolution, by default 1
    bias : bool, optional
        Enable bias computation, by default True
    init_criterion : str, optional
        Weight initialization type, by default 'he'
    weight_init : str, optional
        Weight initialization function type, by default 'quaternion'
    seed : int, optional
        Seed number for Random generator, by default None
    operation : str, optional
        Type of covolution operations, by default 'convolution2d'
    rotation : bool, optional
        Enable quaternion rotation, by default False
    quaternion_format : bool, optional
        Return in quaternion format, by default True
    scale : bool, optional
        Enable quaternoin scaling, by default False
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 dilatation: int = 1, padding: int = 0, groups: int = 1, bias: bool = True, init_criterion: str = 'glorot',
                 weight_init: str = 'quaternion', seed: Optional[str] = None, operation: str = 'convolution2d', rotation: bool = False, quaternion_format: bool = True, scale: bool = False):
        super().__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': hpc_init.quaternion_init,
                      'unitary': hpc_init.unitary_init,
                      'random': hpc_init.random_init}[self.weight_init]
        self.scale = scale

        (self.kernel_size, self.w_shape) = hpc_init.get_kernel_and_weight_shape(self.operation,
                                                                                self.in_channels, self.out_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        hpc_ops.affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                 self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.rotation:
            return hpc_ops.quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                                                    self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                                                    self.quaternion_format, self.scale_param)
        else:
            return hpc_ops.quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                           self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', bias=' + str(self.bias is not None) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) \
            + ', rotation=' + str(self.rotation) \
            + ', q_format=' + str(self.quaternion_format) \
            + ', operation=' + str(self.operation) + ')'


class QuaternionLinearAutograd(Module):
    """Linear Quaternion Layer with custom Autograd.

    Autograd function is call to drastically reduce the VRAM consumption.
    Nonetheless, computing time is also slower compared to QuaternionLinear().

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    bias : bool, optional
        Enable bias computation, by default True
    init_criterion : str, optional
        Weight initialization type, by default 'he'
    weight_init : str, optional
        Weight initialization function type, by default 'quaternion'
    seed : int, optional
        Seed number for Random generator, by default None
    rotation : bool, optional
        Enable quaternion rotation, by default False
    quaternion_format : bool, optional
        Return in quaternion format, by default True
    scale : bool, optional
        Enable quaternoin scaling, by default False
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=True, scale=False):
        super().__init__()
        self.in_features = in_features//4
        self.out_features = out_features//4
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale = scale

        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters using custom initialization."""
        winit = {'quaternion': hpc_init.quaternion_init, 'unitary': hpc_init.unitary_init,
                 'random': hpc_init.random_init}[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        hpc_init.affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                             self.rng, self.init_criterion)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        if self.rotation:
            return hpc_ops.quaternion_linear_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, self.scale_param)
        else:
            return hpc_ops.quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self) -> str:
        """Get str format of layer parameters."""
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', rotation=' + str(self.rotation) \
            + ', seed=' + str(self.seed) + ')'


class QuaternionLinear(Module):
    """Linear Quaternion Layer.

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    bias : bool, optional
        Enable bias computation, by default True
    init_criterion : str, optional
        Weight initialization type, by default 'he'
    weight_init : str, optional
        Weight initialization function type, by default 'quaternion'
    seed : int, optional
        Seed number for Random generator, by default None
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init_criterion: str = 'he', weight_init: str = 'quaternion',
                 seed: Optional[int] = None):
        super().__init__()
        self.in_features = in_features//4
        self.out_features = out_features//4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters using kaiming uniform."""
        winit = {'quaternion': hpc_init.quaternion_init,
                 'unitary': hpc_init.unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        hpc_init.affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                             self.rng, self.init_criterion)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward computation."""
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = hpc_ops.QuaternionLinearFunction.apply(
                input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = hpc_ops.QuaternionLinearFunction.apply(
                input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self) -> str:
        """Get str format of layer parameters."""
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'
