import cupy as cp

from enum import Enum


# Y_hat is model output, and Y is the real label/output
# The expected dimensions are k * m, where k in no of neurons in the output, and m is the batchsize

# https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/metrics/_classification.py#L2176
def cross_entropy_loss(Y_hat, Y):
    m = Y_hat.shape[1]
    xp = cp.get_array_module(Y_hat)

    L_sum = xp.sum(xp.multiply(Y, xp.log(Y_hat)))
    L = -(1.0 / m) * L_sum
    return L


def cross_entropy_derivative(Y_hat, Y):
    # xp = cp.get_array_module(Y_hat)
    # return -(xp.divide(Y, Y_hat) - xp.divide(1 - Y, 1 - Y_hat))
    return Y_hat - Y


def mse(Y_hat, Y):
    # Error is summed across the output neurons
    return 0.5 * ((Y_hat - Y) ** 2).sum(axis=0)


def mse_derivative(Y_hat, Y):
    return Y_hat - Y


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
