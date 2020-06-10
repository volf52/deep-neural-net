import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_FUNCTIONS, ACTIVATION_DERIVATIVES
from mlpcode.activation import ActivationFuncs as af
from mlpcode.callbacks import Callback


class Layer:
    def __init__(self, layerUnits: int, gpu=False):
        self.layerUnits = layerUnits
        if gpu:
            self.xp = cp
        else:
            self.xp = np

        self.gpu = gpu
        self.layerUnits = layerUnits
        self.cache = {}
        self.isBuilt = False

    def build(self):
        self.isBuilt = True


class BatchNormLayer(Layer):
    EPSILON = 1e-2
    MA_BETA = 0.9  # Moving average beta parameter

    def __init__(self, layerUnits: int, gpu=False):
        super(BatchNormLayer, self).__init__(layerUnits, gpu=gpu)
        self.beta = None
        self.gamma = None
        self.mu = None
        self.sigma = None

    @property
    def parameter_shape(self):
        return (self.layerUnits,)

    @property
    def batchNormParams(self):
        assert self.isBuilt
        return (self.gamma, self.beta, self.mu, self.sigma)

    def loadBatchNormParams(
        self, gamma: np.ndarray, beta: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ):
        paramsShape = self.parameter_shape
        assert gamma.shape == paramsShape
        assert beta.shape == paramsShape
        assert mu.shape == paramsShape
        assert sigma.shape == paramsShape

        self.gamma = gamma
        self.beta = beta
        self.mu = mu
        self.sigma = sigma

    def build(self):
        # Beta and mu use zeros init, while gamma and sigma use ones init

        if self.beta is None:
            self.beta = self.xp.zeros(self.layerUnits, dtype=np.float32)
        if self.gamma is None:
            self.gamma = self.xp.ones(self.layerUnits, dtype=np.float32)
        if self.mu is None:
            self.mu = self.xp.zeros(self.layerUnits, dtype=np.float32)
        if self.sigma is None:
            self.sigma = self.xp.ones(self.layerUnits, dtype=np.float32)

        super(BatchNormLayer, self).build()

    def forward(self, Z: np.ndarray, isTrain=True) -> np.ndarray:
        # May add moving average later
        assert self.isBuilt

        self.cache.clear()

        if isTrain:
            mu = Z.mean(axis=0)
            var = Z.var(axis=0)

            zNorm = (Z - mu) / self.xp.sqrt(var + self.EPSILON)

            ztilde = self.gamma * zNorm + self.beta

            self.cache["input"] = Z
            self.cache["znorm"] = zNorm
            self.cache["mu"] = mu
            self.cache["var"] = var
            self.cache["output"] = ztilde
        else:
            ztilde = Z - self.mu
            ztilde /= self.xp.sqrt(self.sigma + self.EPSILON)
            ztilde *= self.gamma
            ztilde += self.beta

        return ztilde

    def backwards(self, delta: np.ndarray, lr: float):
        n, d = self.cache["input"].shape

        xMu = self.cache["input"] - self.cache["mu"]
        stdInv = 1.0 / self.xp.sqrt(self.cache["var"] + self.EPSILON)

        dZnorm = delta * self.gamma
        dVar = -0.5 * self.xp.sum(dZnorm * xMu, axis=0) * stdInv ** 3
        dMu = self.xp.sum(dZnorm * -stdInv, axis=0) + dVar * self.xp.mean(
            -2.0 * xMu, axis=0
        )

        newDelta = (dZnorm * stdInv) + (dVar * 2 * xMu / n) + (dMu / n)
        dGamma = self.xp.sum(delta * self.cache["znorm"], axis=0)
        dBeta = self.xp.sum(delta, axis=0)

        self.gamma -= lr * dGamma
        self.beta -= lr * dBeta

        self.mu = (self.MA_BETA * self.mu) + ((1 - self.MA_BETA) * self.cache["mu"])
        self.sigma = (self.MA_BETA * self.sigma) + (
            (1 - self.MA_BETA) * self.cache["var"]
        )

        self.cache.clear()

        return newDelta


class LinearLayer(Layer):
    def __init__(
        self,
        layerUnits: int,
        inputUnits: int,
        useBias=False,
        gpu=False,
        batchNorm=False,
    ):
        super(LinearLayer, self).__init__(layerUnits, gpu=gpu)

        self.inputUnits = inputUnits
        self.useBias = useBias
        self.weights = None
        self.bias = None
        self.activation = None
        self.batchNorm = batchNorm
        # Change callbacks to nested dict for test and train, with separate callbacks for weights, z, activations
        self.callbacks = []

        if batchNorm:
            self.batchNormLayer = BatchNormLayer(layerUnits, gpu=gpu)

    @property
    def weights_shape(self):
        return self.inputUnits, self.layerUnits

    @property
    def bias_shape(self):
        return (self.layerUnits,)

    @property
    def batchNormParams(self):
        assert self.batchNorm
        return self.batchNormLayer.batchNormParams

    def load_parameters(self, weights: np.ndarray, bias: np.ndarray = None):
        assert weights.shape == self.weights_shape
        self.weights = weights

        if self.useBias:
            assert bias is not None
            assert bias.shape == self.bias_shape
            self.bias = bias

    def loadBatchNormParams(
        self, gamma: np.ndarray, beta: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ):
        assert self.batchNorm

        self.batchNormLayer.loadBatchNormParams(gamma, beta, mu, sigma)

    def addCallback(self, callback: Callback):
        self.callbacks.append(callback)

    def build(self, activation: af = None):

        if self.weights is None:
            self.weights = (
                self.xp.random.randn(self.inputUnits, self.layerUnits)
                * self.xp.sqrt(1 / self.inputUnits)
            ).astype(np.float32)
        if self.useBias and self.bias is None:
            self.bias = self.xp.random.randn(self.layerUnits).astype(np.float32)
        if self.gpu:
            cp.cuda.Stream.null.synchronize()

        self.activation = activation
        if activation is not None:
            assert activation in ACTIVATION_FUNCTIONS

        if self.batchNorm:
            self.batchNormLayer.build()

        super(LinearLayer, self).build()

    def _forward(
        self, X: np.ndarray, weight: np.ndarray, bias: np.ndarray, isTrain=True
    ) -> np.ndarray:
        self.cache.clear()
        self.cache["input"] = X

        # Change weights with callbacks only at test time
        if not isTrain:
            for cb in self.callbacks:
                weight = cb(weight, gpu=self.gpu)

        z = X.dot(weight)
        if bias is not None:
            z += bias

        if self.gpu:
            cp.cuda.Stream.null.synchronize()

        if self.batchNorm:
            z = self.batchNormLayer.forward(z, isTrain=isTrain)

        self.cache["z"] = z
        if self.activation is not None:
            z = ACTIVATION_FUNCTIONS[self.activation](z)
            self.cache["a"] = z

        if not isTrain:
            self.cache.clear()

        return z

    def forward(self, X: np.ndarray, isTrain=True) -> np.ndarray:
        assert self.isBuilt

        z = self._forward(X, self.weights, self.bias, isTrain=isTrain)

        return z

    def backwards(self, dA: np.ndarray, lr: float, activDeriv=True) -> np.ndarray:
        assert "input" in self.cache
        assert "z" in self.cache

        delta = dA  # shape: (num_instances, self.layersUnits_
        if self.activation is not None and activDeriv:
            assert "a" in self.cache
            delta = ACTIVATION_DERIVATIVES[self.activation](dA, self.cache["a"])

        # batchnorm updates to delta
        if self.batchNorm:
            delta = self.batchNormLayer.backwards(delta, lr=lr)

        dw = self.cache["input"].T.dot(delta)

        if self.useBias:
            db = delta.mean(axis=0, dtype=np.float32)
            self.bias -= lr * db

        dlPrev = delta.dot(self.weights.T)

        self.weights -= lr * dw

        self.cache.clear()
        return dlPrev


class BinaryLayer(LinearLayer):
    def __init__(
        self,
        layerUnits: int,
        inputUnits: int,
        useBias=False,
        gpu=False,
        batchNorm=False,
    ):
        super(BinaryLayer, self).__init__(
            layerUnits, inputUnits, useBias=useBias, gpu=gpu, batchNorm=batchNorm,
        )
        self.H = None

    def build(self, activation: af = None):
        if self.H is None:
            self.H = self.xp.sqrt(1.5 / (self.layerUnits + self.inputUnits)).astype(
                np.float32
            )
        super(BinaryLayer, self).build(activation)

    @staticmethod
    def binarize(x: np.ndarray, H=1.0) -> np.ndarray:
        newX = x.copy()
        newX[x >= 0] = 1
        newX[x < 0] = -1
        newX *= H

        return newX

    def forward(self, X: np.ndarray, isTrain=True) -> np.ndarray:
        assert self.isBuilt

        weight = self.binarize(self.weights, H=self.H)
        bias = None
        if self.bias is not None:
            bias = self.binarize(self.bias, H=self.H)

        z = self._forward(X, weight, bias, isTrain=isTrain)
        return z


if __name__ == "__main__":
    z = np.random.randn(10, 10)
    layer = BatchNormLayer(10)
    layer.build()
    print(layer.forward(z))
