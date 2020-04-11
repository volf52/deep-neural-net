from enum import Enum

import cupy as cp

RELU_EPSILON = 0.01


def sigmoid(x):
    # xp allows a generic interface for cpu/gpu code
    xp = cp.get_array_module(x)
    return 1.0 / (1.0 + xp.exp(-x))


def sigmoid_derivate(dA, x):
    z = sigmoid(x)
    return z * (1 - z)


def tanh(x):
    xp = cp.get_array_module(x)
    return xp.tanh(x)


def tanh_derivative(dA, x):
    return 1 - tanh(x) ** 2


def softmax(x):
    xp = cp.get_array_module(x)
    # Subtracting the max to stabilise it (preventing ops with xp.Inf)
    e_x = xp.exp(x - x.max(axis=0, keepdims=True))
    # Sum on axis 0 gives the number of classes
    return e_x / e_x.sum(axis=0, keepdims=True)


def softmax_derivative(dA, x):
    z = softmax(x)
    return z * (1 - z)


def relu(x):
    A = x.copy()
    A[x <= 0] *= RELU_EPSILON
    return A


def relu_derivative(dA, x):
    dZ = dA.copy()
    dZ[x > 0] = 1
    dZ[x <= 0] = RELU_EPSILON
    return dZ


def unitstep(x):
    xp = cp.get_array_module(x)
    return xp.sign(x)


def hard_tanh(dA, x):
    return x.clip(x, -1, 1)


class ActivationFuncs(Enum):
    sigmoid = "sigmoid"
    softmax = "softmax"
    relu = "relu"
    tanh = "tanh"
    sign = "unitstep"

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
}

ACTIVATION_DERIVATIVES = {
    ActivationFuncs.relu: relu_derivative,
    ActivationFuncs.sigmoid: sigmoid_derivate,
    ActivationFuncs.softmax: softmax_derivative,
    ActivationFuncs.tanh: tanh_derivative,
    ActivationFuncs.sign: hard_tanh,
}
