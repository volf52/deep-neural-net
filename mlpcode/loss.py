from enum import Enum

import cupy as cp


# Y_hat is model output, and Y is the real label/output
# The expected dimensions are k * m, where k in no of neurons in the output, and m is the batchsize

# https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/metrics/_classification.py#L2176
def cross_entropy_loss(ypred: cp.ndarray, y: cp.ndarray, eps=1e-15) -> cp.ndarray:
    """
    Cross-entropy loss. Must be called with output layer activation being sigmoid or softmax

    Parameters
    ----------
    ypred
        Numpy array with predicted labels, with shape (num_instances, num_classes) and dtype: float32
    y
        Numpy array with actual labels, with shape (num_instances, num_classes) and dtype: uint8
    eps
        Tolerance for datapoints near zero (to prevent instability with log)

    Returns
    -------
    ndarray
        Scalar array containing the error/cost with size num_instances

    """

    xp = cp.get_array_module(ypred)

    xp.clip(ypred, eps, 1 - eps, out=ypred)

    L = -(y * xp.log(ypred)).sum(axis=1)

    return L


def cross_entropy_derivative(
    ypred: cp.ndarray, y: cp.ndarray, eps=1e-15, with_softmax=False
) -> cp.ndarray:
    """
    Derivative for Mean Squared Error

    Parameters
    ----------
    ypred
        Numpy array with predicted labels, with shape (num_instances, num_classes) and dtype: float32
    y
        Numpy array with true labels, with shape (num_instances, num_classes) and dtype: uint8

    Returns
    -------
    ndarray
        The derivative of cross_entropy_loss(ypred, y) with shape (num_instances, num_classes)
    """

    if with_softmax:
        dA = ypred - y

    else:
        xp = cp.get_array_module(ypred)
        xp.clip(ypred, eps, 1 - eps, out=ypred)
        dA = -(xp.divide(y, ypred) - xp.divide(1 - y, 1 - ypred))

    dA /= ypred.shape[0]

    return dA


def mse(ypred: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    """
    Mean Squared Error

    Parameters
    ----------
    ypred
        Numpy array with predicted labels, with shape (num_instances, num_classes) and dtype: float32
    y
        Numpy array with true labels, with shape (num_instances, num_classes) and dtype: uint8

    Returns
    -------
    ndarray
        Scalar array containing the error/cost with size num_instances
    """

    cost = 0.5 * ((ypred - y) ** 2).sum(axis=1)

    return cost


def mse_derivative(ypred: cp.ndarray, y: cp.ndarray, **kwargs) -> cp.ndarray:
    """
    Derivative for Mean Squared Error

    Parameters
    ----------
    ypred
        Numpy array with predicted labels, with shape (num_instances, num_classes) and dtype: float32
    y
        Numpy array with true labels, with shape (num_instances, num_classes) and dtype: uint8

    Returns
    -------
    ndarray
        The derivative of mse(ypred, y) with shape (num_instances, num_classes)
    """

    return (ypred - y) / y.shape[0]


def hinge_loss(ypred: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    assert ypred.shape == y.shape

    xp = cp.get_array_module(ypred)
    loss = 1 - ypred * y
    xp.maximum(0, loss, out=loss)

    return loss.sum(axis=1)


def hinge_derivative(ypred: cp.ndarray, y: cp.ndarray, **kwargs) -> cp.ndarray:
    assert ypred.shape == y.shape
    assert y.ndim == 2

    xp = cp.get_array_module(ypred)
    v = ypred * y
    grad = xp.where(v < 1, -y / y.shape[0], 0)

    return grad


class LossFuncs(Enum):
    cross_entropy = "cross_entropy"
    mse = "mse"
    hinge = "hinge"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


LOSS_FUNCS = {
    LossFuncs.cross_entropy: cross_entropy_loss,
    LossFuncs.mse: mse,
    LossFuncs.hinge: hinge_loss,
}

LOSS_DERIVATES = {
    LossFuncs.cross_entropy: cross_entropy_derivative,
    LossFuncs.mse: mse_derivative,
    LossFuncs.hinge: hinge_derivative,
}
