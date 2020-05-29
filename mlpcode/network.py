from datetime import datetime
from pathlib import Path
from typing import List, Union, Tuple

import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.optim import LRScheduler
from mlpcode.utils import MODELDIR, XY_DATA

from mlpcode.layers import BinaryLayer, LinearLayer

ArrayList = List[np.ndarray]
ModelLayer = Union[LinearLayer, BinaryLayer]


class Network(object):
    def __init__(
        self, units: List[int], useGpu=False, binarized=False, useBias=False,
    ):

        if useGpu:
            self.xp = cp
        else:
            self.xp = np

        self.isBinarized = binarized
        self.useBias = useBias
        self.useGpu = useGpu
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
            layer(lu, iu, gpu=self.useGpu, useBias=self.useBias)
            for lu, iu in zip(units[1:], units[:-1])
        ]
        return layers

    @property
    def weights(self):
        return [l.weights for l in self._layers]

    @property
    def biases(self):
        return [l.bias for l in self._layers]

    def load_weights(self, weights: ArrayList, biases: ArrayList = None):
        if not self.useBias:
            biases = [None for _ in weights]

        assert len(biases) == len(weights)

        for layer, w, b in zip(self._layers, weights, biases):
            layer.load_weights(w, b)

    @staticmethod
    def fromModel(filePth: Path, useGpu=False, binarized=False, useBias=False):
        assert filePth.exists()
        print(f"\nLoading model from {filePth}\n")
        nn = Network([], useGpu=useGpu, binarized=binarized, useBias=useBias)
        # If the file was saved using cupy, it would convert the weights (and biases)
        # list to an object array, so allow_pickle and subsequent conversion is for that
        with nn.xp.load(filePth, allow_pickle=True) as fp:
            # If the file has been loaded using cupy, there is an extra layer to go through
            if hasattr(fp, "files"):
                npzfile = fp
            else:
                npzfile = fp.npz_file
            # Weights, biases and units
            keyArr = npzfile.files
            assert len(keyArr) == 3
            # Conversion done for the same reason allow_pickle is used above

            weights = [nn.xp.array(x, dtype=np.float32) for x in npzfile[keyArr[0]]]
            biases = [nn.xp.array(x, dtype=np.float32) for x in npzfile[keyArr[1]]]
            units = list(map(int, npzfile[keyArr[2]]))
            nn._layers = nn.__unitsToLayers(units)

            for layer, w, b in zip(nn._layers, weights, biases):
                layer.load_weights(w, b)

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

        best_weights: ArrayList = [0 for _ in self._layers]
        best_biases: ArrayList = best_weights[:]
        best_accuracy = -1.0

        n = len(trainX)

        costList: List[float] = []
        accList: List[float] = []

        if valX is None:
            n_test = n
            valX = trainX.copy()
            valY = trainY.copy()
            accType = "Training"
        else:
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
        for j in range(epochs):
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
            correct = self.evaluate(valX, valY)
            acc = correct * 100.0 / n_test

            accList.append(acc)
            costList.append(cost)

            print(
                "Epoch {0} / {7}:\t{1} Acc: {2} / {3} ({4:.05f}%)\t{5} Loss: {6:.02f}".format(
                    j + 1, accType, correct, n_test, acc, accType, cost, epochs,
                )
            )

            if save_best_params and acc > best_accuracy:
                best_accuracy = acc
                best_weights = [w.copy() for w in self.weights]
                if self.useBias:
                    best_biases = [b.copy() for b in self.biases]
                if self.useGpu:
                    cp.cuda.Stream.null.synchronize()

        if save_best_params:
            print(
                "\nBest {0} Accuracy:\t{1:.03f}%".format(accType, float(best_accuracy))
            )
            print("Switching to best params\n")
            self.load_weights(best_weights, best_biases)

        return costList, accList

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
            a = layer.forward(a, cache=isTraining)

        return a

    def predict_classes(self, X: np.ndarray):
        yhat = self.predict(X, isTraining=False)
        preds = yhat.argmax(axis=1)
        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size=1):
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

    def get_accuracy(self, X: np.ndarray, y: np.ndarray, batch_size=1):
        correct = self.evaluate(X, y, batch_size=batch_size)
        acc = correct * 100.0 / X.shape[0]
        return acc

    def save_weights(self, modelName: str, binarized=False):
        raise NotImplementedError("Give me some time")
        fName = f"{modelName}_{datetime.utcnow().timestamp()}"

        if binarized:
            fName += "_binarized"
        filePth = MODELDIR / fName

        biases = self.biases
        if binarized:
            weights = (self.binarize(w, H=h) for w, h in zip(self.weights, self.H))
            if self.useBias:
                biases = (self.binarize(b, H=h) for b, h in zip(self.biases, self.H))
        else:
            weights = self.weights

        if not self.useBias:
            biases = [0 for _ in biases]

        weights_to_save = [cp.asnumpy(w) for w in weights]
        biases_to_save = [cp.asnumpy(b) for b in biases]
        print(f"Saving model to {filePth}.npz")

        self.xp.savez(filePth, weights_to_save, biases_to_save, self.unitList)

    @staticmethod
    def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int, n: int):
        if batch_size == 1:
            batches = [(X, y)]
        else:
            batches = (
                (X[k : k + batch_size], y[k : k + batch_size])
                for k in range(0, n, batch_size)
            )

        return batches
