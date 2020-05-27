import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_FUNCTIONS, ACTIVATION_DERIVATIVES
from mlpcode.activation import ActivationFuncs as af


class LinearLayer:
    def __init__(
        self,
        layerUnits: int,
        inputUnits: int,
        activation: af = None,
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
        self.activation = activation
        self.weights = None
        self.bias = None
        self.batchNorm = batchNorm
        self.cache = {}
        self.isBuilt = False

        if activation is not None:
            assert activation in ACTIVATION_FUNCTIONS
            self.af = ACTIVATION_FUNCTIONS[activation]
            self.afderiv = ACTIVATION_DERIVATIVES[activation]

    def build(self):
        self.weights = (
            self.xp.random.randn(self.layerUnits, self.inputUnits)
            * self.xp.sqrt(1 / self.inputUnits)
        ).astype(np.float32)
        if self.useBias:
            self.bias = self.xp.random.randn(self.layerUnits, 1).astype(np.float32)
        if self.gpu:
            cp.cuda.Stream.null.synchronize()

        self.isBuilt = True

    def _forward(
        self, X: np.ndarray, weight: np.ndarray, bias: np.ndarray, train=True
    ) -> np.ndarray:
        self.cache.clear()
        self.cache["input"] = X

        z = self.xp.dot(X, weight)
        if bias is not None:
            z += bias
        self.cache["z"] = z

        if self.gpu:
            cp.cuda.Stream.null.synchronize()

        if self.batchNorm:
            pass

        if self.activation is not None:
            z = self.af(z)
            self.cache["a"] = z

        if not train:
            self.cache.clear()

        return z

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert self.isBuilt

        z = self._forward(X, self.weights, self.bias)

        return z

    def backwards(self, dL: np.ndarray, lr: float, lossXEntr=False) -> np.ndarray:
        assert "input" in self.cache
        assert "z" in self.cache

        delta = dL  # shape: (num_instances, self.layersUnits)
        n = delta.shape[0]
        if self.activation is not None:
            assert "a" in self.cache
            if not (lossXEntr and self.activation in (af.sigmoid, af.softmax)):
                delta = dL * self.afderiv(self.cache["a"])

        # batchnorm updates to delta

        dw = np.dot(self.cache["input"].T, delta)
        dw /= n

        self.weights -= lr * dw

        dlPrev = np.dot(delta, self.weights.T)

        self.cache.clear()
        return dlPrev


class BinaryLayer(LinearLayer):
    def __init__(
        self,
        layerUnits: int,
        inputUnits: int,
        activation: af = None,
        useBias=False,
        gpu=False,
        batchNorm=False,
    ):
        assert activation in (None, af.sign, af.identity)
        super(BinaryLayer, self).__init__(
            layerUnits,
            inputUnits,
            activation=activation,
            useBias=useBias,
            gpu=gpu,
            batchNorm=batchNorm,
        )
        self.H = None

    def build(self):
        super(BinaryLayer, self).build()
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

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert self.isBuilt

        weight = self.binarize(self.weights, H=self.H)
        bias = None
        if self.bias is not None:
            bias = self.binarize(self.bias, H=self.H)

        z = self._forward(X, weight, bias)
        return z


def accuracy(trainX, trainY, layers):
    a = trainX
    for layer in layers:
        a = layer.forward(a)

    ypred = np.argmax(a, axis=1)
    correct = (ypred == trainY.squeeze()).mean()
    return correct * 100.0


if __name__ == "__main__":
    from mlpcode.utils import loadDataset, DATASETS
    from mlpcode.loss import LossFuncs as lf
    from mlpcode.loss import LOSS_FUNCS, LOSS_DERIVATES
    from mlpcode.network import Network

    useGpu = False
    lr = 1e-3
    loss = lf.cross_entropy
    lossF = LOSS_FUNCS[loss]
    lossDeriv = LOSS_DERIVATES[loss]

    lossXEntr = loss == lf.cross_entropy

    dataset = DATASETS.mnist
    trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu)
    trainY = np.squeeze(trainY)

    l1 = LinearLayer(784, 256, activation=af.sigmoid, gpu=useGpu)
    l2 = LinearLayer(256, 10, activation=af.softmax, gpu=useGpu)
    layers = [l1, l2]

    for layer in layers:
        layer.build()

    for epoch in range(100):
        epochLoss = []
        for batchX, batchY in Network.get_batches(trainX, trainY, 600, trainX.shape[0]):
            a = batchX
            for layer in layers:
                a = layer.forward(a)

            err = lossF(a, batchY).mean()
            # print("Error:\t{0:.02f}".format(err))
            epochLoss.append(err)

            dl = lossDeriv(a, batchY)

            for layer in reversed(layers):
                dl = layer.backwards(
                    dl, lr, lossXEntr=lossXEntr and layer == layers[-1]
                )

        # print(sum(epochLoss) / len(epochLoss))
        print(accuracy(testX, testY, layers))
