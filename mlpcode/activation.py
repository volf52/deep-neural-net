from enum import Enum

import cupy as cp
from numpy import ndarray

RELU_EPSILON = 0.01

# TODO: Might have to change all a's to z's and use z from cache during backpropogation, for the hard_tanh part


def identity(x: ndarray):
    return x


def identity_derivative(dA: ndarray, a: ndarray):
    delta = dA.copy()
    return delta


def sigmoid(x: ndarray):
    # xp allows a generic interface for cpu/gpu code
    xp = cp.get_array_module(x)
    a = 1.0 / (1.0 + xp.exp(-x))

    return a


def sigmoid_derivate(dA: ndarray, a: ndarray):
    dZ = a * (1 - a)
    delta = dA * dZ
    return delta


def tanh(x: ndarray):
    xp = cp.get_array_module(x)
    a = xp.tanh(x)

    return a


def tanh_derivative(dA: ndarray, a: ndarray):
    dZ = 1 - a ** 2
    delta = dA * dZ
    return delta


def softmax(x: ndarray):
    xp = cp.get_array_module(x)
    # Subtracting the max to stabilise it (preventing ops with xp.Inf)
    a = x - x.max(axis=1)[:, xp.newaxis]
    xp.exp(a, out=a)

    # Sum on axis 1 the total per instance prediction
    a /= a.sum(axis=1)[:, xp.newaxis]

    return a


def softmax_derivative(dA: ndarray, a: ndarray):
    xp = cp.get_array_module(a)
    m, n = a.shape
    a = softmax(a)

    tensor1 = xp.einsum("ij,ik->ijk", a, a)
    tensor2 = xp.einsum("ij,jk->ijk", a, xp.eye(n, n))
    dZ = tensor2 - tensor1

    delta = xp.einsum("ijk,ik->ij", dZ, dA)
    return delta


def relu(x: ndarray):
    a = x.copy()
    a[x < 0] = 0

    return a


def relu_derivative(dA: ndarray, a: ndarray):
    delta = dA.copy()

    delta[a <= 0] = 0

    return delta


def leaky_relu(x: ndarray):
    a = x.copy()
    a[x < 0] *= RELU_EPSILON

    return a


def leaky_relu_derivative(dA: ndarray, a: ndarray):
    delta = dA.copy()

    delta[a <= 0] *= RELU_EPSILON

    return delta


def hard_sigmoid(x: ndarray):
    xp = cp.get_array_module(x)
    a = xp.clip((x + 1.0) / 2, 0.0, 1.0)

    return a


def unitstep(x: ndarray):
    # Faster than a = xp.ones_like(x, dtype=cp.int8); a[x < 0] = -1
    a = x.copy()

    a[x >= 0] = 1
    a[x < 0] = -1

    return a


def hard_tanh(dA: ndarray, a: ndarray):
    # equivalent to max(-1, min(z, 1))
    a = a.clip(-1, 1)
    # No need for the hard_sigmoid thing. Same as clip
    dZ = a * dA
    return dZ


class ActivationFuncs(Enum):
    sigmoid = "sigmoid"
    softmax = "softmax"
    relu = "relu"
    tanh = "tanh"
    sign = "unitstep"
    leaky_relu = "leaky_relu"
    identity = "identity"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


ACTIVATION_FUNCTIONS = {
    ActivationFuncs.sigmoid: sigmoid,
    ActivationFuncs.softmax: softmax,
    ActivationFuncs.relu: relu,
    ActivationFuncs.tanh: tanh,
    ActivationFuncs.sign: unitstep,
    ActivationFuncs.leaky_relu: leaky_relu,
    ActivationFuncs.identity: identity,
}

ACTIVATION_DERIVATIVES = {
    ActivationFuncs.relu: relu_derivative,
    ActivationFuncs.sigmoid: sigmoid_derivate,
    ActivationFuncs.softmax: softmax_derivative,
    ActivationFuncs.tanh: tanh_derivative,
    ActivationFuncs.sign: hard_tanh,
    ActivationFuncs.leaky_relu: leaky_relu_derivative,
    ActivationFuncs.identity: identity_derivative,
}
