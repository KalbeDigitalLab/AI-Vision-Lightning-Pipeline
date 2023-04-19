# source: https://github.com/ispamm/PHBreast/blob/main/models/hypercomplex_ops.py
##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def check_input(input: torch.Tensor):
    """Check input dimension before use.

    Parameters
    ----------
    input : torch.Tensor
        Input quaternion tensor

    Raises
    ------
    RuntimeError
        Quaternion Tensor has more than 5 dimensions.
    RuntimeError
        Quanternion Tensor size is not divisible by 4.
    """

    if input.dim() not in {2, 3, 4, 5}:
        raise RuntimeError(
            'Quaternion linear accepts only input of dimension 2 or 3. Quaternion conv accepts up to 5 dim '
            ' input.dim = ' + str(input.dim())
        )

    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]

    if nb_hidden % 4 != 0:
        raise RuntimeError(
            'Quaternion Tensors must be divisible by 4.'
            ' input.size()[1] = ' + str(nb_hidden)
        )


def get_r(input: torch.Tensor) -> torch.Tensor:
    """Get Real-value from Quaternion Tensor."""
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]

    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, 0, nb_hidden // 4)


def get_i(input: torch.Tensor) -> torch.Tensor:
    """Get 1st Imaginary-value from Quaternion Tensor."""

    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)


def get_j(input: torch.Tensor) -> torch.Tensor:
    """Get 2nd Imaginary-value from Quaternion Tensor."""
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)


def get_k(input: torch.Tensor) -> torch.Tensor:
    """Get 3th Imaginary-value from Quaternion Tensor."""
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)


def q_normalize(input: torch.Tensor, channel: int = 1) -> torch.Tensor:
    """Compute the normalized Quaternion Tensor."""

    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)

    norm = torch.sqrt(r*r + i*i + j*j + k*k + 0.0001)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm

    return torch.cat([r, i, j, k], dim=channel)


def get_modulus(input: torch.Tensor, vector_form: bool = False) -> torch.Tensor:
    """Compute the modulus of Quaternion Tensor."""
    check_input(input)
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)
    if vector_form:
        return torch.sqrt(r * r + i * i + j * j + k * k)
    else:
        return torch.sqrt((r * r + i * i + j * j + k * k).sum(dim=0))


def get_normalized(input: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """Compute the normalized Quaternion Tensor."""
    check_input(input)
    data_modulus = get_modulus(input)
    if input.dim() == 2:
        data_modulus_repeated = data_modulus.repeat(1, 4)
    elif input.dim() == 3:
        data_modulus_repeated = data_modulus.repeat(1, 1, 4)
    return input / (data_modulus_repeated.expand_as(input) + eps)


def quaternion_exp(input: torch.Tensor) -> torch.Tensor:
    """Compute the Exponential form of Quaternion Tensor."""
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)

    norm_v = torch.sqrt(i*i+j*j+k*k) + 0.0001
    exp = torch.exp(r)

    r = torch.cos(norm_v)
    i = (i / norm_v) * torch.sin(norm_v)
    j = (j / norm_v) * torch.sin(norm_v)
    k = (k / norm_v) * torch.sin(norm_v)

    return torch.cat([exp*r, exp*i, exp*j, exp*k], dim=1)


def hamilton_product(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Compute the Hamilton product between two Quaternions."""

    q1_r = get_r(q1)
    q1_i = get_i(q1)
    q1_j = get_j(q1)
    q1_k = get_k(q1)

    # rr', xx', yy', and zz'
    r_base = torch.mul(q0, q1)
    # (rr' - xx' - yy' - zz')
    r = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

    # rx', xr', yz', and zy'
    i_base = torch.mul(q0, torch.cat([q1_i, q1_r, q1_k, q1_j], dim=1))
    # (rx' + xr' + yz' - zy')
    i = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

    # ry', xz', yr', and zx'
    j_base = torch.mul(q0, torch.cat([q1_j, q1_k, q1_r, q1_i], dim=1))
    # (rx' + xr' + yz' - zy')
    j = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

    # rz', xy', yx', and zr'
    k_base = torch.mul(q0, torch.cat([q1_k, q1_j, q1_i, q1_r], dim=1))
    # (rx' + xr' + yz' - zy')
    k = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

    return torch.cat([r, i, j, k], dim=1)


def kronecker_conv(input: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                   stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, groups: int = 1, dilatation: Union[int, Tuple] = 1, learn_A: bool = False, cuda: bool = True, first_layer: bool = False):  # ,
    # mat1_learn, mat2_learn, mat3_learn, mat4_learn):
    """Applies a quaternion convolution using kronecker product."""

    # Define the initial matrices to build the Hamilton product
    if first_layer:
        mat1 = torch.zeros((4, 4), requires_grad=False).view(4, 4, 1, 1)
    else:
        mat1 = torch.eye(4, requires_grad=False).view(4, 4, 1, 1)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]], requires_grad=False).view(4, 4, 1, 1)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]], requires_grad=False).view(4, 4, 1, 1)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]], requires_grad=False).view(4, 4, 1, 1)

    if cuda:
        mat1, mat2, mat3, mat4 = mat1.cuda(), mat2.cuda(), mat3.cuda(), mat4.cuda()

    # Sum of kronecker product between the four matrices and the learnable weights.
    cat_kernels_4_quaternion = torch.kron(mat1, r_weight) + \
        torch.kron(mat2, i_weight) + \
        torch.kron(mat3, j_weight) + \
        torch.kron(mat4, k_weight)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception('The convolutional input is either 3, 4 or 5 dimensions.'
                        ' input.dim = ' + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, dilatation, groups)


def quaternion_conv(input: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                    stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, groups: int = 1, dilatation: Union[int, Tuple] = 1):
    """Applies a quaternion convolution."""

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=1)

    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception('The convolutional input is either 3, 4 or 5 dimensions.'
                        ' input.dim = ' + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, dilatation, groups)


def quaternion_transpose_conv(input: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                              stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, output_padding: Union[int, Tuple] = 0, groups: int = 1, dilatation: Union[int, Tuple] = 1):
    """Applies a quaternion transposed convolution."""

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=1)
    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)

    if input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception('The convolutional input is either 3, 4 or 5 dimensions.'
                        ' input.dim = ' + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, output_padding, groups, dilatation)


def quaternion_conv_rotation(input: torch.Tensor, zero_kernel: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                             stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, groups: int = 1, dilatation: Union[int, Tuple] = 1, quaternion_format: bool = True, scale: Optional[torch.Tensor] = None):
    """Applies a quaternion rotation and convolution transformation.

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Works for unitary and non unitary weights.
    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r = (r_weight*r_weight)
    square_i = (i_weight*i_weight)
    square_j = (j_weight*j_weight)
    square_k = (k_weight*k_weight)

    norm = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    # print(norm)

    r_n_weight = (r_weight / norm)
    i_n_weight = (i_weight / norm)
    j_n_weight = (j_weight / norm)
    k_n_weight = (k_weight / norm)

    norm_factor = 2.0

    square_i = norm_factor*(i_n_weight*i_n_weight)
    square_j = norm_factor*(j_n_weight*j_n_weight)
    square_k = norm_factor*(k_n_weight*k_n_weight)

    ri = (norm_factor*r_n_weight*i_n_weight)
    rj = (norm_factor*r_n_weight*j_n_weight)
    rk = (norm_factor*r_n_weight*k_n_weight)

    ij = (norm_factor*i_n_weight*j_n_weight)
    ik = (norm_factor*i_n_weight*k_n_weight)

    jk = (norm_factor*j_n_weight*k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat([zero_kernel, scale * (1.0 - (square_j + square_k)),
                                     scale * (ij-rk), scale * (ik+rj)], dim=1)
            rot_kernel_2 = torch.cat([zero_kernel, scale * (ij+rk), scale *
                                     (1.0 - (square_i + square_k)), scale * (jk-ri)], dim=1)
            rot_kernel_3 = torch.cat([zero_kernel, scale * (ik-rj), scale * (jk+ri),
                                     scale * (1.0 - (square_i + square_j))], dim=1)
        else:
            rot_kernel_1 = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij-rk), (ik+rj)], dim=1)
            rot_kernel_2 = torch.cat([zero_kernel, (ij+rk), (1.0 - (square_i + square_k)), (jk-ri)], dim=1)
            rot_kernel_3 = torch.cat([zero_kernel, (ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=1)

        zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=1)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    else:
        if scale is not None:
            rot_kernel_1 = torch.cat([scale * (1.0 - (square_j + square_k)),
                                     scale * (ij-rk), scale * (ik+rj)], dim=0)
            rot_kernel_2 = torch.cat(
                [scale * (ij+rk), scale * (1.0 - (square_i + square_k)), scale * (jk-ri)], dim=0)
            rot_kernel_3 = torch.cat([scale * (ik-rj), scale * (jk+ri), scale *
                                     (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2 = torch.cat([(ij+rk), 1.0 - (square_i + square_k), (jk-ri)], dim=0)
            rot_kernel_3 = torch.cat([(ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    # print(input.shape)
    # print(square_r.shape)
    # print(global_rot_kernel.shape)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception('The convolutional input is either 3, 4 or 5 dimensions.'
                        ' input.dim = ' + str(input.dim()))

    return convfunc(input, global_rot_kernel, bias, stride, padding, dilatation, groups)


def quaternion_transpose_conv_rotation(input: torch.Tensor, zero_kernel: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                                       stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, output_padding: Union[int, Tuple] = 0, groups: int = 1, dilatation: Union[int, Tuple] = 1, quaternion_format: bool = True):
    """Applies a quaternion rotation and transposed convolution transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Works for unitary and non unitary weights.
    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r = (r_weight*r_weight)
    square_i = (i_weight*i_weight)
    square_j = (j_weight*j_weight)
    square_k = (k_weight*k_weight)

    norm = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    r_weight = (r_weight / norm)
    i_weight = (i_weight / norm)
    j_weight = (j_weight / norm)
    k_weight = (k_weight / norm)

    norm_factor = 2.0

    square_i = norm_factor*(i_weight*i_weight)
    square_j = norm_factor*(j_weight*j_weight)
    square_k = norm_factor*(k_weight*k_weight)

    ri = (norm_factor*r_weight*i_weight)
    rj = (norm_factor*r_weight*j_weight)
    rk = (norm_factor*r_weight*k_weight)

    ij = (norm_factor*i_weight*j_weight)
    ik = (norm_factor*i_weight*k_weight)

    jk = (norm_factor*j_weight*k_weight)

    if quaternion_format:
        rot_kernel_1 = torch.cat([zero_kernel, 1.0 - (square_j + square_k), ij-rk, ik+rj], dim=1)
        rot_kernel_2 = torch.cat([zero_kernel, ij+rk, 1.0 - (square_i + square_k), jk-ri], dim=1)
        rot_kernel_3 = torch.cat([zero_kernel, ik-rj, jk+ri, 1.0 - (square_i + square_j)], dim=1)

        zero_kernel2 = torch.zeros(rot_kernel_1.shape).cuda()
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)
    else:
        rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), ij-rk, ik+rj], dim=1)
        rot_kernel_2 = torch.cat([ij+rk, 1.0 - (square_i + square_k), jk-ri], dim=1)
        rot_kernel_3 = torch.cat([ik-rj, jk+ri, 1.0 - (square_i + square_j)], dim=1)
        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    if input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception('The convolutional input is either 3, 4 or 5 dimensions.'
                        ' input.dim = ' + str(input.dim()))

    return convfunc(input, global_rot_kernel, bias, stride, padding, output_padding, groups, dilatation)


def quaternion_linear(input: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
    """Applies a quaternion linear transformation.

    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_4_quaternion is a modified version of the quaternion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.
    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)

    if input.dim() == 2:

        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output+bias
        else:
            return output


def quaternion_linear_rotation(input: torch.Tensor, zero_kernel: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                               quaternion_format: bool = False, scale: bool = None) -> torch.Tensor:
    """Applies a quaternion rotation transformation.

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Works for unitary and non unitary weights.
    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r = (r_weight*r_weight)
    square_i = (i_weight*i_weight)
    square_j = (j_weight*j_weight)
    square_k = (k_weight*k_weight)

    norm = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    r_n_weight = (r_weight / norm)
    i_n_weight = (i_weight / norm)
    j_n_weight = (j_weight / norm)
    k_n_weight = (k_weight / norm)

    norm_factor = 2.0

    square_i = norm_factor*(i_n_weight*i_n_weight)
    square_j = norm_factor*(j_n_weight*j_n_weight)
    square_k = norm_factor*(k_n_weight*k_n_weight)

    ri = (norm_factor*r_n_weight*i_n_weight)
    rj = (norm_factor*r_n_weight*j_n_weight)
    rk = (norm_factor*r_n_weight*k_n_weight)

    ij = (norm_factor*i_n_weight*j_n_weight)
    ik = (norm_factor*i_n_weight*k_n_weight)

    jk = (norm_factor*j_n_weight*k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat([zero_kernel, scale * (1.0 - (square_j + square_k)),
                                     scale * (ij-rk), scale * (ik+rj)], dim=0)
            rot_kernel_2 = torch.cat([zero_kernel, scale * (ij+rk), scale *
                                     (1.0 - (square_i + square_k)), scale * (jk-ri)], dim=0)
            rot_kernel_3 = torch.cat([zero_kernel, scale * (ik-rj), scale * (jk+ri),
                                     scale * (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2 = torch.cat([zero_kernel, (ij+rk), (1.0 - (square_i + square_k)), (jk-ri)], dim=0)
            rot_kernel_3 = torch.cat([zero_kernel, (ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=0)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)

    else:
        if scale is not None:
            rot_kernel_1 = torch.cat([scale * (1.0 - (square_j + square_k)),
                                     scale * (ij-rk), scale * (ik+rj)], dim=0)
            rot_kernel_2 = torch.cat(
                [scale * (ij+rk), scale * (1.0 - (square_i + square_k)), scale * (jk-ri)], dim=0)
            rot_kernel_3 = torch.cat([scale * (ik-rj), scale * (jk+ri), scale *
                                     (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2 = torch.cat([(ij+rk), 1.0 - (square_i + square_k), (jk-ri)], dim=0)
            rot_kernel_3 = torch.cat([(ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)

    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, global_rot_kernel)
        else:
            return torch.mm(input, global_rot_kernel)
    else:
        output = torch.matmul(input, global_rot_kernel)
        if bias is not None:
            return output+bias
        else:
            return output


class QuaternionLinearFunction(torch.autograd.Function):
    """Customized the Autograd computation for Linear Quaaternion Functions.

    Custom function to reduce the VRAM consumption.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Compute the forward propagation."""
        ctx.save_for_backward(input, r_weight, i_weight, j_weight, k_weight, bias)
        check_input(input)
        cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
        cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
        cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion = torch.cat(
            [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
        if input.dim() == 2:
            if bias is not None:
                return torch.addmm(bias, input, cat_kernels_4_quaternion)
            else:
                return torch.mm(input, cat_kernels_4_quaternion)
        else:
            output = torch.matmul(input, cat_kernels_4_quaternion)
            if bias is not None:
                return output+bias
            else:
                return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        """Compute the backward propagation."""

        input, r_weight, i_weight, j_weight, k_weight, bias = ctx.saved_tensors
        grad_input = grad_weight_r = grad_weight_i = grad_weight_j = grad_weight_k = grad_bias = None

        input_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        input_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
        input_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
        input_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion_T = Variable(
            torch.cat([input_r, input_i, input_j, input_k], dim=1).permute(1, 0), requires_grad=False)

        r = get_r(input)
        i = get_i(input)
        j = get_j(input)
        k = get_k(input)
        input_r = torch.cat([r, -i, -j, -k], dim=0)
        input_i = torch.cat([i,  r, -k, j], dim=0)
        input_j = torch.cat([j,  k, r, -i], dim=0)
        input_k = torch.cat([k,  -j, i, r], dim=0)
        input_mat = Variable(torch.cat([input_r, input_i, input_j, input_k], dim=1), requires_grad=False)

        r = get_r(grad_output)
        i = get_i(grad_output)
        j = get_j(grad_output)
        k = get_k(grad_output)
        input_r = torch.cat([r, i, j, k], dim=1)
        input_i = torch.cat([-i,  r, k, -j], dim=1)
        input_j = torch.cat([-j,  -k, r, i], dim=1)
        input_k = torch.cat([-k,  j, -i, r], dim=1)
        grad_mat = torch.cat([input_r, input_i, input_j, input_k], dim=0)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(cat_kernels_4_quaternion_T)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_mat.permute(1, 0).mm(input_mat).permute(1, 0)
            unit_size_x = r_weight.size(0)
            unit_size_y = r_weight.size(1)
            grad_weight_r = grad_weight.narrow(0, 0, unit_size_x).narrow(1, 0, unit_size_y)
            grad_weight_i = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y, unit_size_y)
            grad_weight_j = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y*2, unit_size_y)
            grad_weight_k = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y*3, unit_size_y)
        if ctx.needs_input_grad[5]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias
