from enum import Enum

import cupy as cp
from numpy import ndarray

RELU_EPSILON = 0.01


def identity(x: ndarray):
    return x


def identity_derivative(z: ndarray):
    x = cp.get_array_module(z).ones_like(z)
    return x


def sigmoid(x: ndarray):
    # xp allows a generic interface for cpu/gpu code
    xp = cp.get_array_module(x)
    a = 1.0 / (1.0 + xp.exp(-x))

    return a


def sigmoid_derivate(z: ndarray):
    a = sigmoid(z)
    x = a * (1 - a)

    return x


def tanh(x: ndarray):
    xp = cp.get_array_module(x)
    a = xp.tanh(x)

    return a


def tanh_derivative(z: ndarray):
    xp = cp.get_array_module(z)
    x = 1 - xp.tanh(z) ** 2

    return x


def softmax(x: ndarray):
    xp = cp.get_array_module(x)
    # Subtracting the max to stabilise it (preventing ops with xp.Inf)
    a = x - x.max(axis=0)[xp.newaxis, :]
    xp.exp(a, out=a)

    # Sum on axis 0 gives the number of classes
    a /= a.sum(axis=0)[xp.newaxis, :]

    return a


def softmax_derivative(z: ndarray):
    a = softmax(z)
    x = a * (1 - a)

    return x


def relu(x: ndarray):
    a = x.copy()
    a[x < 0] = 0

    return a


def relu_derivative(z: ndarray):
    x = z.copy()

    x[z <= 0] = 0
    x[z > 0] = 1

    return x


def leaky_relu(x: ndarray):
    a = x.copy()
    a[x < 0] *= RELU_EPSILON

    return a


def leaky_relu_derivative(z: ndarray):
    x = z.copy()

    x[z > 0] = 1
    x[z <= 0] = RELU_EPSILON

    return x


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


def hard_tanh(z: ndarray):
    # equivalent to max(-1, min(z, 1))
    a = z.clip(-1, 1)
    # a = 2 * hard_sigmoid(z) - 1

    return a


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
