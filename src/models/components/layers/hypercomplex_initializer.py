# source: https://github.com/ispamm/PHBreast/blob/main/models/hypercomplex_ops.py
##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

#
# PARAMETERS INITIALIZATION
#

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from numpy.random import RandomState
from scipy.stats import chi
from torch.autograd import Variable


def unitary_init(in_features: int, out_features: int, rng: np.random.RandomState, kernel_size=None, criterion: str = 'he'):
    """Initialize Quaternion Tensor as Unitary Quaternion."""
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field  # noqa F841
        fan_out = out_features * receptive_field  # noqa F841
    else:
        fan_in = in_features  # noqa F841
        fan_out = out_features  # noqa F841

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
        v_r[i] /= norm
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)


def random_init(in_features: int, out_features: int, rng: np.random.RandomState, kernel_size: Optional[Tuple[int, int]] = None, criterion: str = 'glorot'):
    """Initialize Quaternion Tensor with uniform distribution."""
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))  # noqa F841
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)  # noqa F841
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r
    weight_i = v_i
    weight_j = v_j
    weight_k = v_k
    return (weight_r, weight_i, weight_j, weight_k)


def quaternion_init(in_features: int, out_features: int, rng: np.random.RandomState, kernel_size: Optional[Tuple[int, int]] = None, criterion: str = 'glorot'):
    """Initialize Quaternion Tensor from rng function."""
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1, 1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 + 0.0001)
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def create_dropout_mask(dropout_p: float, size: Union[int, Tuple], rng: np.random.RandomState, as_type: Any, operation: str = 'linear'):
    """Create dropout mask for Linear Quaterion Tensor."""
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1-dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
        raise Exception("create_dropout_mask accepts only 'linear'. Found operation = "
                        + str(operation))


def affect_init(r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, init_func: Callable, rng: np.random.RandomState, init_criterion: str):
    """Modify Quaternion Linear Parameters."""
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
            r_weight.size() != k_weight.size():
        raise ValueError('The real and imaginary weights '
                         'should have the same size . Found: r:'
                         + str(r_weight.size()) + ' i:'
                         + str(i_weight.size()) + ' j:'
                         + str(j_weight.size()) + ' k:'
                         + str(k_weight.size()))

    elif r_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(r_weight.dim()))
    kernel_size = None
    r, i, j, k = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def affect_init_conv(r_weight: torch.Tensor, i_weight: torch.Tensor, j_weight: torch.Tensor, k_weight: torch.Tensor, kernel_size: Tuple, init_func: Callable, rng: np.random.RandomState,
                     init_criterion: str):
    """Modify Quaternion Convolution Parameters."""
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
            r_weight.size() != k_weight.size():
        raise ValueError('The real and imaginary weights '
                         'should have the same size . Found: r:'
                         + str(r_weight.size()) + ' i:'
                         + str(i_weight.size()) + ' j:'
                         + str(j_weight.size()) + ' k:'
                         + str(k_weight.size()))

    elif 2 >= r_weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(r_weight.dim()))

    r, i, j, k = init_func(
        r_weight.size(1),
        r_weight.size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion
    )
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def get_kernel_and_weight_shape(operation: str, in_channels: int, out_channels: int, kernel_size: int) -> Tuple:
    """Get kernel and weight shapes."""
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:  # in case it is 2d or 3d.
        if operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape
