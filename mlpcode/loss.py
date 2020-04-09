import cupy as cp
import numpy as np

from enum import Enum


# Y_hat is model output, and Y is the real label/output


def cross_entropy_loss(Y_hat, Y):
    m = Y_hat.shape[1]
    xp = cp.get_array_module(Y_hat)

    Y_hat[xp.isclose(Y_hat, 0.0000001)] += 2.2251e-308

    cost = (
        -1.0
        / m
        * (xp.dot(Y, xp.log(Y_hat).T) + xp.dot(1 - Y, xp.log(1 - Y_hat).T))
    )
    return xp.squeeze(cost)


def check_zero_one(arr):
    xp = cp.get_array_module(arr)
    return xp.logical_or(xp.equal(arr, 0), xp.equal(arr, 1))


def cross_entropy_derivative(Y_hat, Y):
    xp = cp.get_array_module(Y_hat)

    Y_hat[check_zero_one(Y_hat)] += 0000000000.1

    return -(xp.divide(Y, Y_hat) - xp.divide(1 - Y, 1 - Y_hat))


def mse(Y_hat, Y):
    return 0.5 * ((Y - Y_hat) ** 2)


def mse_derivative(Y_hat, Y):
    return Y - Y_hat


class LossFuncs(Enum):
    cross_entropy = "cross_entropy"
    mse = "mse"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


LOSS_FUNCS = {
    LossFuncs.cross_entropy: cross_entropy_loss,
    LossFuncs.mse: mse,
}

LOSS_DERIVATES = {
    LossFuncs.cross_entropy: cross_entropy_derivative,
    LossFuncs.mse: mse_derivative,
}
