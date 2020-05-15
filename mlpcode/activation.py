from enum import Enum

import cupy as cp


RELU_EPSILON = 0.01

# TODO Add support for inplace derivatives
# TODO Cast to proper datatypes to lower memory footprint


def identity(x):
    return x


def identity_derivative(dA, z):
    return cp.get_array_module(z).ones_like(z)


def sigmoid(x):
    "x should be of the shape (n_features, n_samples)"
    # xp allows a generic interface for cpu/gpu code
    xp = cp.get_array_module(x)
    return 1.0 / (1.0 + xp.exp(-x))


def sigmoid_derivate(dA, z):
    a = sigmoid(z)
    return a * (1 - a)


def tanh(x):
    "x should be of the shape (n_features, n_samples)"
    xp = cp.get_array_module(x)
    return xp.tanh(x, out=x)


def tanh_derivative(dA, z):
    xp = cp.get_array_module(z)
    return 1 - xp.tanh(z) ** 2


def softmax(x):
    "x should be of the shape (n_features, n_samples)"
    xp = cp.get_array_module(x)
    # Subtracting the max to stabilise it (preventing ops with xp.Inf)
    tmp = x - x.max(axis=0)[xp.newaxis, :]
    xp.exp(tmp, out=x)
    # Sum on axis 0 gives the number of classes
    x /= x.sum(axis=0)[xp.newaxis, :]

    return x


def softmax_derivative(dA, z):
    a = softmax(z)
    return a * (1 - a)


def relu(x):
    "x should be of the shape (n_features, n_samples)"
    xp = cp.get_array_module(x)
    xp.clip(x, 0, xp.finfo(x.dtype).max, out=x)
    return x


def relu_derivative(dA, z):
    dZ = dA.copy()
    dZ[z <= 0] = 0
    dZ[z > 0] = 1
    return dZ


def leaky_relu(x):
    "x should be of the shape (n_features, n_samples)"
    A = x.copy()
    A[x <= 0] *= RELU_EPSILON
    return A


def leaky_relu_derivative(dA, z):
    dZ = dA.copy()
    dZ[z > 0] = 1
    dZ[z <= 0] = RELU_EPSILON
    return dZ


def hard_sigmoid(x):
    xp = cp.get_array_module(x)
    return xp.clip((x + 1.0) / 2, 0.0, 1.0)


def unitstep(x):
    "x should be of the shape (n_features, n_samples)"
    x[x >= 0] = 1
    x[x < 0] = -1
    return x


def hard_tanh(dA, z):
    # equivalent to max(-1, min(z, 1))
    return z.clip(-1, 1)
    # return 2 * hard_sigmoid(z) - 1


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
