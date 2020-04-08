from enum import Enum
from math import exp

import cupy as cp
import numpy as np


def sigmoid(x):
    # xp allows a generic interface for cpu/gpu code
    xp = cp.get_array_module(x)
    return 1 / (1 + xp.exp(-x))


def sigmoid_derivate(dA, x):
    z = sigmoid(x)
    return dA * z * (1 - z)


def tanh(x):
    xp = cp.get_array_module(x)
    return xp.tanh(x)


def tanh_derivative(dA, x):
    return 1 - tanh(x) ** 2


def softmax(x):
    xp = cp.get_array_module(x)
    # Subtracting the max to stabilise it (preventing ops with xp.Inf)
    e_x = xp.exp(x - xp.max(x))
    return e_x / xp.sum(e_x, axis=0)


def softmax_derivative(dA, x):
    z = softmax(x)
    return dA * z * (1 - z)


def relu(x):
    xp = cp.get_array_module(x)
    return xp.maximum(0, x)


def relu_derivative(dA, x):
    xp = cp.get_array_module(x)
    dZ = xp.array(dA, copy=True)
    dZ[x <= 0] = 0
    return dZ


def unitstep(x):
    xp = cp.get_array_module(x)
    return xp.sign(x)


def hard_tanh(x):
    xp = cp.get_array_module(x)
    return xp.clip(x, -1, 1)


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


if __name__ == "__main__":
    import time

    size = int(1e7)
    nptestArr = np.linspace(-size, size, dtype=np.float32)
    cpTestArr = cp.linspace(-size, size, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()

    s = time.time()
    a = hard_tanh(nptestArr)
    e = time.time()
    print(f"Time for numpy:\t\t{e - s}")

    s = time.time()
    b = hard_tanh(cpTestArr)
    e = time.time()
    print(f"Time for cupy:\t\t{e - s}")
    print(np.max(a))
    print(cp.max(b))
