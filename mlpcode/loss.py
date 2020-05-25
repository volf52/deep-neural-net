from enum import Enum

import cupy as cp


# Y_hat is model output, and Y is the real label/output
# The expected dimensions are k * m, where k in no of neurons in the output, and m is the batchsize

# https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/metrics/_classification.py#L2176
def cross_entropy_loss(yhat, y, eps=1e-15):
    """
    Cross-entropy loss. Must be called with output layer activation being sigmoid or softmax

    Parameters
    ----------
    yhat
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: float32
    y
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: uint8
    eps
        Tolerance for datapoints near zero (to prevent instability with log)

    Returns
    -------
    ndarray
        Scalar array containing the error/cost
    """

    m = yhat.shape[1]
    xp = cp.get_array_module(yhat)

    xp.clip(yhat, eps, 1 - eps, out=yhat)

    L = -(y * xp.log(yhat)).sum() / m

    return L


def cross_entropy_derivative(yhat, y):
    """
    Derivative for Mean Squared Error

    Parameters
    ----------
    yhat
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: float32
    y
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: uint8

    Returns
    -------
    ndarray
        The derivative of cross_entropy_loss(yhat, y) with shape(num_classes, num_instances)
    """

    # xp = cp.get_array_module(Y_hat)
    # cost = -(xp.divide(Y, Y_hat) - xp.divide(1 - Y, 1 - Y_hat))
    cost = yhat - y

    return cost


def mse(yhat, y):
    """
    Mean Squared Error

    Parameters
    ----------
    yhat
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: float32
    y
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: uint8

    Returns
    -------
    ndarray
        Scalar array containing the error/cost
    """

    cost = 0.5 * ((yhat - y) ** 2).sum(axis=0)

    return cost


def mse_derivative(yhat, y):
    """
    Derivative for Mean Squared Error

    Parameters
    ----------
    yhat
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: float32
    y
        Numpy array with predicted labels, with shape (num_classes, num_instances) and dtype: uint8

    Returns
    -------
    ndarray
        The derivative of mse(yhat, y) with shape(num_classes, num_instances)
    """

    return yhat - y


def hinge_loss(yhat: cp.ndarray, y: cp.ndarray):
    assert yhat.shape == y.shape

    xp = cp.get_array_module(yhat)
    loss = 1 - yhat * y
    xp.maximum(0, loss, out=loss)

    return loss.mean(axis=0)


def hinge_derivative(yhat: cp.ndarray, y: cp.ndarray):
    assert yhat.shape == y.shape
    assert y.ndim == 2

    xp = cp.get_array_module(yhat)
    v = yhat * y
    grad = xp.where(v < 1, -y / y.shape[0], 0)

    return grad


# def squared_hinge_loss(yhat: cp.ndarray, y: cp.ndarray):
#     assert yhat.shape == y.shape
#
#     xp = cp.get_array_module(yhat)
#     loss = 1 - yhat * y
#     loss[loss < 0] = 0
#     xp.square(loss, out=loss)
#
#     return loss.mean(axis=0)


# def squared_hinge_derivative(yhat: cp.ndarray, y: cp.ndarray):
#     assert yhat.shape == y.shape
#     assert y.ndim == 2
#
#     grad = 2 * hinge_derivative(yhat, y)
#
#     return grad


class LossFuncs(Enum):
    cross_entropy = "cross_entropy"
    mse = "mse"
    hinge = "hinge"
    # sq_hinge = "squared_hinge"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


LOSS_FUNCS = {
    LossFuncs.cross_entropy: cross_entropy_loss,
    LossFuncs.mse: mse,
    LossFuncs.hinge: hinge_loss,
    # LossFuncs.sq_hinge: squared_hinge_loss,
}

LOSS_DERIVATES = {
    LossFuncs.cross_entropy: cross_entropy_derivative,
    LossFuncs.mse: mse_derivative,
    LossFuncs.hinge: hinge_derivative,
    # LossFuncs.sq_hinge: squared_hinge_derivative,
}
