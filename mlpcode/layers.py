import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_FUNCTIONS, ACTIVATION_DERIVATIVES
from mlpcode.activation import ActivationFuncs as af


class LinearLayer:
    def __init__(
        self,
        layerUnits: int,
        inputUnits: int,
        useBias=False,
        gpu=False,
        batchNorm=False,
    ):
        if gpu:
            self.xp = cp
        else:
            self.xp = np

        self.gpu = gpu
        self.layerUnits = layerUnits
        self.inputUnits = inputUnits
        self.useBias = useBias
        self.weights = None
        self.bias = None
        self.activation = None
        self.batchNorm = batchNorm
        self.cache = {}
        self.isBuilt = False

    @property
    def weights_shape(self):
        return self.inputUnits, self.layerUnits

    @property
    def bias_shape(self):
        return (self.layerUnits,)

    def load_weights(self, weights: np.ndarray, bias: np.ndarray = None):
        assert weights.shape == self.weights_shape
        self.weights = weights

        if self.useBias:
            assert bias is not None
            assert bias.shape == self.bias_shape
            self.bias = bias

    def build(self, activation: af = None):
        if self.weights is None:
            self.weights = (
                self.xp.random.randn(self.inputUnits, self.layerUnits)
                * self.xp.sqrt(1 / self.inputUnits)
            ).astype(np.float32)
        if self.useBias and self.bias is None:
            self.bias = self.xp.random.randn(self.layerUnits, 1).astype(np.float32)
        if self.gpu:
            cp.cuda.Stream.null.synchronize()

        self.activation = activation
        if activation is not None:
            assert activation in ACTIVATION_FUNCTIONS

        self.isBuilt = True

    def _forward(
        self, X: np.ndarray, weight: np.ndarray, bias: np.ndarray, cache=True
    ) -> np.ndarray:
        self.cache.clear()
        self.cache["input"] = X

        z = X.dot(weight)
        if bias is not None:
            z += bias
        self.cache["z"] = z

        if self.gpu:
            cp.cuda.Stream.null.synchronize()

        if self.batchNorm:
            pass

        if self.activation is not None:
            z = ACTIVATION_FUNCTIONS[self.activation](z)
            self.cache["a"] = z

        if not cache:
            self.cache.clear()

        return z

    def forward(self, X: np.ndarray, cache=True) -> np.ndarray:
        assert self.isBuilt

        z = self._forward(X, self.weights, self.bias, cache=cache)

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
            pass

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
        super(BinaryLayer, self).build(activation)
        self.H = self.xp.sqrt(1.5 / (self.layerUnits + self.inputUnits)).astype(
            np.float32
        )

    @staticmethod
    def binarize(x: np.ndarray, H=1.0) -> np.ndarray:
        newX = x.copy()
        newX[x >= 0] = 1
        newX[x < 0] = -1
        newX *= H

        return newX

    def forward(self, X: np.ndarray, cache=True) -> np.ndarray:
        assert self.isBuilt

        weight = self.binarize(self.weights, H=self.H)
        bias = None
        if self.bias is not None:
            bias = self.binarize(self.bias, H=self.H)

        z = self._forward(X, weight, bias, cache=cache)
        return z


def accuracy(X, y, layers):
    a = X
    for layer in layers:
        a = layer.forward(a, cache=False)

    ypred = np.argmax(a, axis=1)
    correct = (ypred == y.squeeze()).mean()
    return correct * 100.0


if __name__ == "__main__":
    from mlpcode.utils import loadDataset, DATASETS
    from mlpcode.loss import LossFuncs as lf
    from mlpcode.loss import LOSS_FUNCS, LOSS_DERIVATES
    from mlpcode.network import Network

    useGpu = True
    lr = 1e-3
    lr = 0.07
    loss = lf.cross_entropy
    lossF = LOSS_FUNCS[loss]
    lossDeriv = LOSS_DERIVATES[loss]

    dataset = DATASETS.mnist
    trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu, encoded=True)
    # testY = np.squeeze(testY)

    valX = trainX.copy()
    valY = trainY.argmax(axis=1)

    l1 = LinearLayer(512, 784, gpu=useGpu)
    l2 = LinearLayer(10, 512, gpu=useGpu)

    l1.build(af.leaky_relu)
    l2.build(af.softmax)

    layers = [l1, l2]

    softCrossEntropy = loss == lf.cross_entropy and layers[-1].activation in (
        af.softmax,
        af.sigmoid,
    )

    for epoch in range(100):
        epochLoss = []
        for batchX, batchY in Network.get_batches(trainX, trainY, 100, trainX.shape[0]):
            a = batchX
            for layer in layers:
                a = layer.forward(a)

            err = lossF(a, batchY).mean()
            epochLoss.append(err)

            da = lossDeriv(a, batchY, with_softmax=softCrossEntropy)
            da = layers[-1].backwards(da, lr, activDeriv=not softCrossEntropy)

            for layer in reversed(layers[:-1]):
                da = layer.backwards(da, lr)

        # print(sum(epochLoss) / len(epochLoss))
        print(accuracy(valX, valY, layers))

    print("Test acc")
    print(accuracy(testX, testY, layers))
