from datetime import datetime
from pathlib import Path
from typing import List, Union

import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.optim import LRScheduler
from mlpcode.utils import MODELDIR


# TODO: remove option for binarized on Network init. Move it only to training


class Network(object):
    def __init__(
        self, layers: List[int], useGpu=False, fineTuning=False, binarized=False,
    ):

        if useGpu:
            self.xp = cp
        else:
            self.xp = np

        self.layers = layers
        self.isBinarized = binarized
        if self.isBinarized:
            self.non_binarized_weights: List[np.ndarray] = []
            self.non_binarized_biases: List[np.ndarray] = []

        if fineTuning:
            self.weights, self.biases = [], []
        else:
            # Xavier init
            self.biases = [
                self.xp.random.randn(y, 1).astype(np.float32) for y in layers[1:]
            ]
            # Casting to float32 at multiple steps to keep the memory footprint low for each step
            self.weights = [
                (
                    self.xp.random.randn(l, l_minus_1).astype(np.float32)
                    * self.xp.sqrt(1 / l_minus_1)
                ).astype(np.float32)
                for l_minus_1, l in zip(layers[:-1], layers[1:])
            ]
        cp.cuda.Stream.null.synchronize()

        self.__lossF = None
        self.__loss = None
        self.__loss_derivative = None
        self.__hiddenDerivative = None
        self.__outAF = None
        self.__outputDerivative = None
        self.__activations = None
        self.__lr: LRScheduler = None
        self.num_layers = len(layers) - 1
        self.__isCompiled = False

    @property
    def isCompiled(self) -> bool:
        return self.__isCompiled

    @staticmethod
    def fromModel(
        filePth: Path, useGpu=False, binarized=False,
    ):
        assert filePth.exists()
        print(f"\nLoading model from {filePth}\n")
        nn = Network([], useGpu=useGpu, fineTuning=True, binarized=binarized,)
        # If the file was saved using cupy, it would convert the weights (and biases)
        # list to an object array, so allow_pickle and subsequent conversion is for that
        with nn.xp.load(filePth, allow_pickle=True) as fp:
            # If the file has been loaded using cupy, there is an extra layer to go through
            if hasattr(fp, "files"):
                npzfile = fp
            else:
                npzfile = fp.npz_file
            # Weights, biases and layers
            keyArr = npzfile.files
            assert len(keyArr) == 3
            # Conversion done for the same reason allow_pickle is used above
            nn.weights = [nn.xp.array(x, dtype=np.float32) for x in npzfile[keyArr[0]]]
            nn.biases = [nn.xp.array(x, dtype=np.float32) for x in npzfile[keyArr[1]]]
            # Just to ensure consistency
            nn.layers = list(map(int, fp[keyArr[2]]))
            nn.num_layers = len(nn.layers) - 1
        return nn

    @staticmethod
    def binarize(x):
        xp = cp.get_array_module(x)
        newX = xp.empty_like(x, dtype=np.int8)
        newX[x >= 0] = 1
        newX[x < 0] = -1
        cp.cuda.Stream.null.synchronize()
        return newX

    def compile(
        self,
        lr: Union[LRScheduler, float] = 1e-3,
        hiddenAf: af = af.sigmoid,
        outAf: af = af.sigmoid,
        lossF: lf = lf.mse,
    ):
        assert hiddenAf in ACTIVATION_FUNCTIONS
        assert outAf in ACTIVATION_FUNCTIONS
        assert lossF in LOSS_FUNCS
        if lossF == lf.cross_entropy and outAf not in (af.sigmoid, af.softmax):
            # My implementation for normal derivative of cross entropy is not stable enough
            # Luckily, it comes down to (output - target) when used with sigmoid or softmax
            raise ValueError("Gotta use sigmoid or softmax with cross entropy loss")

        self.__lossF = lossF
        self.__loss = LOSS_FUNCS[lossF]
        self.__loss_derivative = LOSS_DERIVATES[lossF]
        self.__hiddenDerivative = ACTIVATION_DERIVATIVES[hiddenAf]
        self.__outAF = outAf
        self.__outputDerivative = ACTIVATION_DERIVATIVES[outAf]

        hiddenActivationFunc = ACTIVATION_FUNCTIONS[hiddenAf]
        outputActivationFunc = ACTIVATION_FUNCTIONS[outAf]
        self.__activations = [hiddenActivationFunc for _ in range(self.num_layers - 1)]
        self.__activations.append(outputActivationFunc)
        assert len(self.__activations) == self.num_layers  # len(layers)

        if isinstance(lr, float):
            self.__lr = LRScheduler(alpha=lr)
        elif isinstance(lr, LRScheduler):
            self.__lr = lr
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
        save_best_params=False,
    ):
        if not self.__isCompiled:
            print("\nEXCEPTION: Must compile the model before running train")
            return

        best_weights = [None for _ in self.layers]
        best_biases = best_weights[:]
        best_accuracy = -1.0

        n = len(trainX)
        costList = []
        accList = []
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

        valY = valY.reshape(-1, 1)
        print(f"\n\nStarting training (binarized: {self.isBinarized})")
        print("=" * 20)
        print()
        for j in range(epochs):
            epochCost = []

            # random shuffling
            if shuffle:
                p = self.xp.random.permutation(n)
                trainX = trainX[p, :]
                trainY = trainY[p, :]

            batches = self.get_batches(trainX, trainY, batch_size, n)
            for batch in batches:
                batchCost = self.__update_batch(batch, self.__lr.value)
                epochCost.append(batchCost)

            # The step could be moved inside the loop above
            # Decay rate is meant to be done once per epoch, but that could very well work for each batch
            self.__lr.step()

            correct = self.evaluate(valX, valY, binarized=self.isBinarized)
            cp.cuda.Stream.null.synchronize()
            acc = correct * 100.0 / n_test
            cost = float(self.xp.array(epochCost).mean())
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
                best_biases = [b.copy() for b in self.biases]
                cp.cuda.Stream.null.synchronize()

        if save_best_params:
            print(
                "\nBest {0} Accuracy:\t{1:.03f}%".format(accType, float(best_accuracy))
            )
            print("Switching to best params\n")
            self.weights = best_weights[:]
            self.biases = best_biases[:]

        return costList, accList

    def __update_batch(self, batch, lr):
        x, y = batch
        m = x.shape[0] * 1.0
        if self.isBinarized:
            weights = [self.binarize(w) for w in self.weights]
            biases = [self.binarize(b) for b in self.biases]
        else:
            weights = self.weights
            biases = self.biases
        delta_nabla_b, delta_nabla_w, cost = self.__backprop(x, y, weights, biases)
        cp.cuda.Stream.null.synchronize()
        # matrix.sum(axis=0) => 3
        # matrix.sum(axis=1) => 6
        # [1,2,3]
        # [1,2,3]
        # [1,2,3]

        # matrix.sum(axis=0) => 3
        # matrix.sum(axis=1) => 10
        # [1,2,3,4]
        # [1,2,3,4]
        # [1,2,3,4]

        nabla_b = [nb.mean(axis=1, keepdims=True) for nb in delta_nabla_b]
        nabla_w = [(nw / m) for nw in delta_nabla_w]
        cp.cuda.Stream.null.synchronize()

        self.weights = [w - (lr * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr * nb) for b, nb in zip(self.biases, nabla_b)]
        cp.cuda.Stream.null.synchronize()
        return cost

    def __backprop(self, x, y, weights, biases):
        nabla_b = [None for _ in biases]
        nabla_w = [None for _ in weights]

        # forward pass
        zs, activations, activation = self.__forwardpass(x, weights, biases)

        y = y.T
        # Mean cost of whole batch
        cost = self.__loss(activation, y).mean()
        # backward pass
        dLdA = self.__loss_derivative(activation, y)  # expected shape: k * n
        cp.cuda.Stream.null.synchronize()
        if (self.__outAF == af.identity) or (
            self.__outAF == af.softmax and self.__lossF == lf.cross_entropy
        ):
            delta = dLdA
        else:
            dAdZ = self.__outputDerivative(dLdA, zs[-1])
            delta = dLdA * dAdZ
        nabla_b[-1] = delta
        nabla_w[-1] = self.xp.dot(delta, activations[-2].T)
        cp.cuda.Stream.null.synchronize()

        for l in range(2, self.num_layers + 1):
            z = zs[-l]
            dAprev = self.xp.dot(weights[-l + 1].T, delta)
            cp.cuda.Stream.null.synchronize()
            delta = dAprev * self.__hiddenDerivative(dAprev, z)
            cp.cuda.Stream.null.synchronize()
            nabla_b[-l] = delta
            nabla_w[-l] = self.xp.dot(delta, activations[-l - 1].T)
            cp.cuda.Stream.null.synchronize()
        return (nabla_b, nabla_w, cost)

    def __forwardpass(self, x, weights, biases):
        a = x.T
        activations = [a]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        # (num_features, num_examples)
        for w, b, afunc in zip(weights, biases, self.__activations):
            z = self.xp.dot(w, a) + b
            cp.cuda.Stream.null.synchronize()
            zs.append(z)
            a = afunc(z)
            activations.append(a)
        return zs, activations, a

    def predict(self, X, weights=None, biases=None):
        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases
        _, _, yhat = self.__forwardpass(X, weights=weights, biases=biases)
        cp.cuda.Stream.null.synchronize()
        preds = yhat.argmax(axis=0).reshape(-1, 1)
        return preds

    def evaluate(self, X, y, batch_size=1, binarized=False):
        # testY should NOT be one hot encoded for this to work
        # The code at the start of training takes care of it if testY was one-hot encoded
        # when passed into the train func

        # Expected shape for X: num_instances * num_features
        # Expected shape for y: num_instances * 1

        if binarized:
            weights = [self.binarize(w) for w in self.weights]
            biases = [self.binarize(b) for b in self.biases]
        else:
            weights = self.weights
            biases = self.biases

        batches = self.get_batches(X, y, batch_size=batch_size, n=X.shape[0])
        correct = 0
        for batchX, batchY in batches:
            preds = self.predict(batchX, weights=weights, biases=biases)
            batch_correct = (batchY == preds).sum()
            correct += batch_correct
        return int(correct)

    def save_weights(self, modelName: str, binarized=False):
        fName = f"{modelName}_{datetime.utcnow().timestamp()}"
        if binarized:
            fName += "_binarized"
        filePth = MODELDIR / fName

        if binarized:
            weights = (self.binarize(w) for w in self.weights)
            biases = (self.binarize(b) for b in self.biases)
        else:
            weights = self.weights
            biases = self.biases

        weights_to_save = [cp.asnumpy(w) for w in weights]
        biases_to_save = [cp.asnumpy(b) for b in biases]
        print(f"Saving model to {filePth}.npz")
        self.xp.savez(filePth, weights_to_save, biases_to_save, self.layers)

    @staticmethod
    def get_batches(X, y, batch_size, n):
        if batch_size == 1:
            batches = [(X, y)]
        else:
            batches = [
                (X[k : k + batch_size, :], y[k : k + batch_size, :],)
                for k in range(0, n, batch_size)
            ]
        return batches
