import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_FUNCTIONS, ACTIVATION_DERIVATIVES
from mlpcode.activation import ActivationFuncs as af


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

    def __init__(self, layerUnits: int, gpu=False):
        super(BatchNormLayer, self).__init__(layerUnits, gpu=gpu)
        self.beta = None
        self.gamma = None

    @property
    def parameter_shape(self):
        return (self.layerUnits,)

    @property
    def batchNormParams(self):
        return (self.gamma, self.beta)

    def loadBatchNormParams(self, gamma: np.ndarray, beta: np.ndarray):
        raise NotImplementedError()

    def build(self):
        self.beta = self.xp.random.randn(self.layerUnits).astype(np.float32)
        self.gamma = self.xp.random.randn(self.layerUnits).astype(np.float32)
        super(BatchNormLayer, self).build()

    def forward(self, z: np.ndarray, isTrain=True) -> np.ndarray:
        # May add moving average later
        assert self.isBuilt

        self.cache.clear()
        self.cache["input"] = z

        mu = z.mean()
        sigma = self.xp.sqrt(z.var() + self.EPSILON)

        zNorm = (z - mu) / sigma

        ztilde = self.beta * zNorm + self.gamma

        self.cache["output"] = ztilde

        if not isTrain:
            self.cache.clear()

        # return ztilde
        return z

    def backwards(self, delta: np.ndarray):

        self.cache.clear()
        return delta


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

    def loadBatchNormParams(self, gamma: np.ndarray, beta: np.ndarray):
        assert self.batchNorm

        self.batchNormLayer.load_parameters(gamma, beta)

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

        delta = dA  # shape: (self.layersUnits, num_instances)
        n = delta.shape[0]
        if self.activation is not None and activDeriv:
            assert "a" in self.cache
            delta = ACTIVATION_DERIVATIVES[self.activation](dA, self.cache["a"])

        # batchnorm updates to delta
        if self.batchNorm:
            delta = self.batchNormLayer.backwards(delta)

        dw = self.cache["input"].T.dot(delta)
        # dw /= n

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
