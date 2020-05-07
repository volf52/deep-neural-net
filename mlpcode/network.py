import cupy as cp
import numpy as np
from pathlib import Path
from datetime import datetime

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af, hard_sigmoid
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.utils import MODELDIR, DATASETS


class Network(object):
    def __init__(
        self,
        layers,
        useGpu=False,
        fineTuning=False,
        binarized=False,
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

        if useGpu:
            self.xp = cp
        else:
            self.xp = np

        self.layers = layers
        self.isBinarized = binarized

        if fineTuning:
            self.weights, self.biases = None, None
        else:
            # Xavier init
            self.biases = [self.xp.random.randn(y, 1) for y in layers[1:]]
            self.weights = [
                (self.xp.random.randn(l, l_minus_1) * self.xp.sqrt(1 / l_minus_1))
                for l_minus_1, l in zip(layers[:-1], layers[1:])
            ]

        cp.cuda.Stream.null.synchronize()
        self.loss = LOSS_FUNCS[lossF]
        self.loss_derivative = LOSS_DERIVATES[lossF]
        self.hiddenDerivative = ACTIVATION_DERIVATIVES[hiddenAf]
        self.outAF = outAf
        self.outputDerivative = ACTIVATION_DERIVATIVES[outAf]
        if not fineTuning:
            # Not counting input layer
            self.num_layers = len(layers) - 1
            hiddenActivationFunc = ACTIVATION_FUNCTIONS[hiddenAf]
            outputActivationFunc = ACTIVATION_FUNCTIONS[outAf]
            self.activations = [
                hiddenActivationFunc for _ in range(self.num_layers - 1)
            ]
            self.activations.append(outputActivationFunc)
            assert len(self.activations) == self.num_layers  # len(layers)

    @staticmethod
    def fromModel(
        filePth: Path,
        useGpu=False,
        binarized=False,
        hiddenAf: af = af.sigmoid,
        outAf: af = af.sigmoid,
        lossF: lf = lf.mse,
    ):
        assert filePth.exists()
        nn = Network(
            [],
            useGpu=useGpu,
            hiddenAf=hiddenAf,
            outAf=outAf,
            lossF=lossF,
            fineTuning=True,
            binarized=binarized,
        )
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
            nn.weights = [nn.xp.array(x, dtype=nn.xp.float) for x in npzfile[keyArr[0]]]
            nn.biases = [nn.xp.array(x, dtype=nn.xp.float) for x in npzfile[keyArr[1]]]
            # Just to ensure consistency
            nn.layers = list(map(int, fp[keyArr[2]]))
            nn.num_layers = len(nn.layers) - 1
            hiddenActivationFunc = ACTIVATION_FUNCTIONS[hiddenAf]
            outputActivationFunc = ACTIVATION_FUNCTIONS[outAf]
            nn.activations = [hiddenActivationFunc for _ in range(nn.num_layers - 1)]
            nn.activations.append(outputActivationFunc)
        return nn

    @staticmethod
    def binarize(x):
        return x

    def feedforward(self, x):
        a = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        # (num_features, num_examples)
        if self.isBinarized:
            zipObj = zip(
                self.binarize(self.weights),
                self.binarize(self.biases),
                self.activations,
            )
        else:
            zipObj = zip(self.weights, self.biases, self.activations)
        for w, b, afunc in zipObj:
            z = self.xp.dot(w, a) + b
            cp.cuda.Stream.null.synchronize()
            zs.append(z)
            a = afunc(z)
            activations.append(a)
        return zs, activations, a

    def train(
        self,
        trainX: np.array,
        trainY: np.array,
        epochs: int,
        batch_size=1,
        lr=1e-3,
        valX=None,
        valY=None,
        save: DATASETS = None,
    ):
        best_weights = [None for _ in self.layers]
        best_biases = best_weights[:]
        best_accuracy = -1.0

        n = len(trainX)
        costList = []
        accList = []
        if valX is None:
            n_test = n
            valX = trainX.copy().T
            valY = trainY.copy()
            accType = "Training"
        else:
            n_test = len(valX)
            valX = valX.T
            accType = "Validation"
        # No need to keep this in hot vector encoded form
        if valY.shape[0] != valY.size:
            valY = valY.argmax(axis=1)
        valY = valY.reshape(1, -1)
        print("Starting training")
        for j in range(epochs):
            # random shuffling
            p = self.xp.random.permutation(n)
            epochCost = []
            trainX = trainX[p, :]
            trainY = trainY[p, :]

            batches = [
                (trainX[k : k + batch_size, :], trainY[k : k + batch_size, :],)
                for k in range(0, n, batch_size)
            ]
            for batch in batches:
                batchCost = self.update_batch(batch, lr)
                epochCost.append(batchCost)

            correct = self.get_accuracy(valX, valY)
            cp.cuda.Stream.null.synchronize()
            acc = correct * 100.0 / n_test
            cost = self.xp.array(epochCost).mean()
            accList.append(acc)
            costList.append(cost)
            print(
                "Epoch {0}:\t{1} Acc: {2} / {3} ({4:.05f}%)\t{5} Loss: {6:.02f}".format(
                    j + 1, accType, correct, n_test, float(acc), accType, float(cost)
                )
            )

            if save is not None and acc > best_accuracy:
                best_accuracy = acc
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]

        print("\nBest Accuracy:\t{0:.03f}%".format(float(best_accuracy)))

        if save is not None:
            fName = f"{str(save)}_{datetime.utcnow().timestamp()}"
            filePth = MODELDIR / fName
            print(f"Saving model to {filePth}.npz")
            self.xp.savez(filePth, best_weights, best_biases, self.layers)
        return costList, accList

    def update_batch(self, batch, eta):
        x, y = batch

        m = x.shape[0] * 1.0

        # if self.isBinarized:
        #     self.weights = [self.binarize(w) for w in self.weights]
        #     self.biases = [self.binarize(b) for b in self.biases]
        #     cp.cuda.Stream.null.synchronize()

        delta_nabla_b, delta_nabla_w, cost = self.backprop(x.T, y.T)
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
        # old_weights = self.weights[:]
        # old_biases = self.biases[:]
        self.weights = [w - (eta * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb) for b, nb in zip(self.biases, nabla_b)]

        cp.cuda.Stream.null.synchronize()
        # if self.isBinarized:
        #     self.weights = [self.xp.clip(w, -1.0, 1.0) for w in self.weights]
        #     self.biases = [self.xp.clip(b, -1.0, 1.0) for b in self.biases]
        #     cp.cuda.Stream.null.synchronize()

        return cost

    def backprop(self, x, y):
        nabla_b = [None for b in self.biases]
        nabla_w = [None for w in self.weights]

        # forward pass
        zs, activations, activation = self.feedforward(x)

        # Mean cost of whole batch
        cost = self.loss(activation, y).mean()
        # backward pass
        dLdA = self.loss_derivative(activation, y)  # expected shape: k * n
        cp.cuda.Stream.null.synchronize()
        if self.outAF in (af.softmax, af.identity):
            delta = dLdA
        else:
            dAdZ = self.outputDerivative(dLdA, zs[-1])
            delta = dLdA * dAdZ
        nabla_b[-1] = delta
        nabla_w[-1] = self.xp.dot(delta, activations[-2].T)
        cp.cuda.Stream.null.synchronize()

        for l in range(2, self.num_layers + 1):
            z = zs[-l]
            dAprev = self.xp.dot(self.weights[-l + 1].T, delta)
            cp.cuda.Stream.null.synchronize()
            delta = dAprev * self.hiddenDerivative(dAprev, z)
            cp.cuda.Stream.null.synchronize()
            nabla_b[-l] = delta
            nabla_w[-l] = self.xp.dot(delta, activations[-l - 1].T)
            cp.cuda.Stream.null.synchronize()
        return (nabla_b, nabla_w, cost)

    def get_accuracy(self, X, y):
        # testY should NOT be one hot encoded for this to work
        # The code at the start of training takes care of it if testY was one-hot encoded
        # when passed into the train func
        _, _, y_hat = self.feedforward(X)
        cp.cuda.Stream.null.synchronize()
        preds = y_hat.argmax(axis=0).reshape(1, -1)
        return (y == preds).sum()
