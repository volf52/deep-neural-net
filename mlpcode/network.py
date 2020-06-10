import h5py
from datetime import datetime
from pathlib import Path
from typing import List, Union

import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.layers import BinaryLayer, LinearLayer, BatchNormLayer
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.optim import LRScheduler
from mlpcode.utils import MODELDIR, XY_DATA
from mlpcode.callbacks import Callback

ArrayList = List[np.ndarray]
ModelLayer = Union[LinearLayer, BinaryLayer, BatchNormLayer]


class Network(object):
    def __init__(
        self,
        units: List[int],
        useGpu=False,
        binarized=False,
        useBias=False,
        useBatchNorm=False,
    ):

        if useGpu:
            self.xp = cp
        else:
            self.xp = np

        if useBatchNorm and useBias:
            print(
                "Won't use bias with BatchNorm, as it will automatically get cancelled in the normalization step"
            )
            useBias = False

        self.isBinarized = binarized
        self.useBias = useBias
        self.useGpu = useGpu
        self.useBatchNorm = useBatchNorm
        self.unitList = units
        self._layers = self.__unitsToLayers(units)

        self._lossF: lf = None
        self._outAf: af = None
        self._lr: LRScheduler = None
        self.__isCompiled = False

    @property
    def isCompiled(self) -> bool:
        return self.__isCompiled

    def __unitsToLayers(self, units: List[int]) -> List[ModelLayer]:
        layer: ModelLayer = BinaryLayer if self.isBinarized else LinearLayer
        layers = [
            layer(
                lu,
                iu,
                gpu=self.useGpu,
                useBias=self.useBias,
                batchNorm=self.useBatchNorm,
            )
            for lu, iu in zip(units[1:], units[:-1])
        ]
        return layers

    @property
    def weights(self):
        return [l.weights for l in self._layers]

    @property
    def biases(self):
        if self.useBias:
            return [l.bias for l in self._layers]
        else:
            return [None for _ in self.unitList[:-1]]

    @property
    def batchNormParams(self):
        assert self.useBatchNorm

        gammas = []
        betas = []
        mus = []
        sigmas = []
        for layer in self._layers:
            gamma, beta, mu, sigma = layer.batchNormParams
            gammas.append(gamma)
            betas.append(beta)
            mus.append(mu)
            sigmas.append(sigma)

        return gammas, betas, mus, sigmas

    def _copyBnParams(self):
        assert self.useBatchNorm
        gammas, betas, mus, sigmas = self.batchNormParams

        gammas = [g.copy() for g in gammas]
        betas = [bts.copy() for bts in betas]
        mus = [mu.copy() for mu in mus]
        sigmas = [sigma.copy() for sigma in sigmas]

        return gammas, betas, mus, sigmas

    @property
    def linearLayers(self):
        return [layer for layer in self._layers if isinstance(layer, LinearLayer)]

    def load_weights(self, weights: ArrayList, biases: ArrayList = None):
        if not self.useBias:
            biases = [None for _ in weights]

        assert len(weights) == len(self._layers)
        assert len(biases) == len(weights)

        for layer, w, b in zip(self.linearLayers, weights, biases):
            layer.load_parameters(w, b)

    def loadBatchNormParameters(
        self, gammas: ArrayList, betas: ArrayList, mus: ArrayList, sigmas: ArrayList
    ):
        assert len(gammas) == len(self._layers)
        assert len(betas) == len(gammas)
        assert len(mus) == len(betas)
        assert len(sigmas) == len(mus)

        for gamma, beta, mu, sigma, layer in zip(
            gammas, betas, mus, sigmas, self._layers
        ):
            layer.loadBatchNormParams(gamma, beta, mu, sigma)

    def addCallbacks(self, callbacks: Union[Callback, List[Callback]], num_layers: int=None):
        if num_layers is None:
            num_layers = len(self._layers)

        assert num_layers > 0
        if isinstance(callbacks, List):
            assert len(callbacks) == num_layers
            assert all(isinstance(cb, Callback) for cb in callbacks)
        else:
            callbacks = [callbacks] * num_layers

        for layer, cb in zip(self._layers[:num_layers], callbacks):
            layer.addCallback(cb)

    @staticmethod
    def fromModel(filePth: Path, useGpu=False, binarized=False):
        assert filePth.exists()
        print(f"\nLoading model from {filePth}\n")

        with h5py.File(filePth, "r") as fp:
            fpKeys = fp.keys()
            assert "units" in fpKeys
            assert "weights" in fpKeys
            assert "useBias" in fpKeys
            assert "useBatchNorm" in fpKeys

            unitList = list(map(int, fp["units"][()]))
            useBias = bool(fp["useBias"][()])
            useBatchNorm = bool(fp["useBatchNorm"][()])
            if useBatchNorm and useBias:
                print("Setting useBias to false cause of batchnorm")
                useBias = False

            nn = Network(
                unitList,
                useGpu=useGpu,
                useBias=useBias,
                binarized=binarized,
                useBatchNorm=useBatchNorm,
            )

            weights = []
            biases = []
            ws = fp["weights"]
            for w in ws.values():
                weights.append(nn.xp.array(w[()], dtype=np.float32))

            if useBias:
                assert "biases" in fpKeys
                bs = fp["biases"]
                for b in bs.values():
                    biases.append(nn.xp.array(b[()], dtype=np.float32))

            nn.load_weights(weights, biases)

            if useBatchNorm:
                assert "gammas" in fpKeys
                assert "betas" in fpKeys
                assert "mus" in fpKeys
                assert "sigmas" in fpKeys

                gammas = []
                betas = []
                mus = []
                sigmas = []

                gs = fp["gammas"]
                bts = fp["betas"]
                muDS = fp["mus"]
                sigmaDS = fp["sigmas"]

                for gamma in gs.values():
                    gammas.append(nn.xp.array(gamma[()], dtype=np.float32))

                for beta in bts.values():
                    betas.append(nn.xp.array(beta[()], dtype=np.float32))

                for mu in muDS.values():
                    mus.append(nn.xp.array(mu[()], dtype=np.float32))

                for sigma in sigmaDS.values():
                    sigmas.append(nn.xp.array(sigma[()], dtype=np.float32))

                nn.loadBatchNormParameters(gammas, betas, mus, sigmas)

        return nn

    def compile(
        self,
        lr: Union[LRScheduler, float] = 1e-3,
        hiddenAf: af = af.sigmoid,
        outAf: af = af.softmax,
        lossF: lf = lf.cross_entropy,
    ) -> None:

        assert hiddenAf in ACTIVATION_FUNCTIONS
        assert outAf in ACTIVATION_FUNCTIONS
        assert lossF in LOSS_FUNCS

        if self.isBinarized and hiddenAf != af.sign:
            print(f"Changing hidden activation function to {af.sign} for BNN")
            hiddenAf = af.sign

        for layer in self._layers[:-1]:
            layer.build(hiddenAf)

        self._layers[-1].build(outAf)

        self._lossF = lossF
        self._outAf = outAf

        if isinstance(lr, float):
            self._lr = LRScheduler(alpha=lr)
        elif isinstance(lr, LRScheduler):
            self._lr = lr
        else:
            raise ValueError(
                "Invalid value for learning rate (only float or LRScheduler allowed)"
            )

        self.__isCompiled = True

    def train(
        self,
        trainX: np.ndarray,
        trainY: np.ndarray,
        epochs: int,
        batch_size=1,
        shuffle=True,
        valX: np.ndarray = None,
        valY: np.ndarray = None,
        save_best_params=True,
    ):
        if not self.__isCompiled:
            print("\nEXCEPTION: Must compile the model before running train")
            return

        best_accuracy = -1.0
        best_acc_epoch = -1
        if save_best_params:
            best_weights: ArrayList = [None for _ in self._layers]
            best_biases: ArrayList = best_weights[:]
            best_bn_params = best_weights[:]

        n = len(trainX)

        costList: List[float] = []
        trainAccList: List[float] = []
        valAccList: List[float] = []

        if valX is not None:
            n_test = len(valX)
            accType = "Validation"

        # No need to keep this in hot vector encoded form
        if valY.shape[0] != valY.size:
            valY = valY.argmax(axis=1).astype(np.uint8)

        # Binary One hot encoded {0, 1} to {-1, 1}
        if self._lossF == lf.hinge:
            trainY[trainY == 0] = -1

        print(f"\n\nStarting training (binarized: {self.isBinarized})")
        print("=" * 20)
        print()
        for curr_epoch in range(epochs):
            epochCost: List[float] = []

            # random shuffling
            if shuffle:
                p = self.xp.random.permutation(n)
                trainX = trainX[p, :]
                trainY = trainY[p, :]

            batches = self.get_batches(trainX, trainY, batch_size, n)
            for batch in batches:
                batchCost = self.__updateBatch(batch, self._lr.value)
                epochCost.append(batchCost)

            # The step could be moved inside the loop above
            # Decay rate is meant to be done once per epoch, but that could very well work for each batch
            self._lr.step()

            cost = sum(epochCost) / len(epochCost)
            trainLabels = trainY.argmax(axis=1).astype(np.uint8)
            acc = self.get_accuracy(trainX, trainLabels)

            trainAccList.append(acc)
            costList.append(cost)

            mainStr = f"Epoch {curr_epoch+1} / {epochs}\tAccuracy : {acc:0.3f}%"

            if valX is not None:
                acc = self.get_accuracy(valX, valY)
                valAccList.append(acc)

            if valX is not None:
                mainStr += f"\tVal Acc: {acc:0.3f}%"
            mainStr += f"\tLoss: {cost:.02f}"
            print(mainStr)

            if save_best_params and acc > best_accuracy:
                best_accuracy = acc
                best_acc_epoch = curr_epoch
                best_weights = [w.copy() for w in self.weights]
                if self.useBias:
                    best_biases = [b.copy() for b in self.biases]
                if self.useBatchNorm:
                    best_bn_params = self._copyBnParams()
                if self.useGpu:
                    cp.cuda.Stream.null.synchronize()

        if save_best_params:
            print(
                "\nBest {0} Accuracy:\t{1:.03f}% (epoch: {2})".format(
                    accType, float(best_accuracy), best_acc_epoch
                )
            )
            print("Switching to best params\n")
            self.load_weights(best_weights, best_biases)
            if self.useBatchNorm:
                self.loadBatchNormParameters(*best_bn_params)

        history = {"loss": costList, "accuracy": trainAccList}
        if valX is not None:
            history["val_accuracy"] = valAccList

        return history

    def __updateBatch(self, batch: XY_DATA, lr: float):
        X, y = batch

        output = self.predict(X, isTraining=True)

        cost = LOSS_FUNCS[self._lossF](output, y)
        cost = float(cost.mean())

        softCrossEntropy = self._lossF == lf.cross_entropy and self._outAf in (
            af.softmax,
            af.sigmoid,
        )

        dLdA = LOSS_DERIVATES[self._lossF](output, y, with_softmax=softCrossEntropy)

        delta = self._layers[-1].backwards(dLdA, lr, activDeriv=not softCrossEntropy)

        for layer in reversed(self._layers[:-1]):
            delta = layer.backwards(delta, lr)

        return cost

    def predict(self, X: np.ndarray, isTraining=False):
        a = X
        for layer in self._layers:
            a = layer.forward(a, isTrain=isTraining)

        return a

    def predict_classes(self, X: np.ndarray):
        yhat = self.predict(X, isTraining=False)
        preds = yhat.argmax(axis=1)
        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size=-1):
        # testY should NOT be one hot encoded for this to work
        # The code at the start of training takes care of it if testY was one-hot encoded
        # when passed into the train func

        # Expected shape for X: num_instances * num_features
        # Expected shape for y: num_instances

        batches = self.get_batches(X, y, batch_size=batch_size, n=X.shape[0])
        correct = 0
        for batchX, batchY in batches:
            preds = self.predict_classes(batchX)
            batch_correct = (batchY == preds).sum()
            correct += batch_correct

        return int(correct)

    def get_accuracy(self, X: np.ndarray, y: np.ndarray, batch_size=-1):
        correct = self.evaluate(X, y, batch_size=batch_size)
        acc = correct * 100.0 / X.shape[0]
        return acc

    def save_weights(self, modelName: str, binarized=False):
        fName = f"{modelName}_{datetime.utcnow().timestamp()}"

        if binarized:
            fName += "_binarized"

        filePth = MODELDIR / f"{fName}.hdf5"

        with h5py.File(filePth, "w") as fp:
            fp.create_dataset("units", data=np.array(self.unitList, dtype=np.uint32))
            fp.create_dataset("useBias", data=self.useBias)
            fp.create_dataset("useBatchNorm", data=self.useBatchNorm)
            ws = fp.create_group("weights")
            for i, w in enumerate(self.weights):
                ws.create_dataset(f"weights_{i}", data=cp.asnumpy(w), dtype=np.float32)

            if self.useBias:
                bs = fp.create_group("biases")
                for i, b in enumerate(self.biases):
                    bs.create_dataset(
                        f"biases_{i}", data=cp.asnumpy(b), dtype=np.float32
                    )

            if self.useBatchNorm:
                gs = fp.create_group("gammas")
                bts = fp.create_group("betas")
                mus = fp.create_group("mus")
                sigmas = fp.create_group("sigmas")
                for i, (gamma, beta, mu, sigma) in enumerate(
                    zip(*self.batchNormParams)
                ):
                    gs.create_dataset(
                        f"gammas_{i}", data=cp.asnumpy(gamma), dtype=np.float32
                    )
                    bts.create_dataset(
                        f"betas_{i}", data=cp.asnumpy(beta), dtype=np.float32
                    )
                    mus.create_dataset(
                        f"mus_{i}", data=cp.asnumpy(mu), dtype=np.float32
                    )
                    sigmas.create_dataset(
                        f"sigmas_{i}", data=cp.asnumpy(sigma), dtype=np.float32
                    )

        print(f"Saving model to {filePth}")

    @staticmethod
    def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int, n: int):
        if batch_size == -1:
            batches = [(X, y)]
        else:
            batches = (
                (X[k : k + batch_size], y[k : k + batch_size])
                for k in range(0, n, batch_size)
            )

        return batches
