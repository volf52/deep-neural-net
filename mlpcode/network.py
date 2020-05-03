import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af, unitstep as binarize
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.utils import saveNpy, loadWeightsBiasesNpy


class Network(object):
    def __init__(
        self,
        sizes,
        useGpu=False,
        fineTuning=False,
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
            raise ValueError(
                "Gotta use sigmoid or softmax with cross entropy loss"
            )

        # Not counting input layer
        self.num_layers = len(sizes) - 1
        if useGpu:
            self.xp = cp
        else:
            self.xp = np
        self.sizes = sizes
        if fineTuning:
            self.weights, self.biases = loadWeightsBiasesNpy(self.xp)
        else:
            self.biases = [self.xp.random.randn(y, 1) for y in sizes[1:]]
            # Xavier init
            self.weights = [
                (
                    self.xp.random.randn(l, l_minus_1)
                    * self.xp.sqrt(1 / l_minus_1)
                )
                for l_minus_1, l in zip(sizes[:-1], sizes[1:])
                ]
        cp.cuda.Stream.null.synchronize()
        self.loss = LOSS_FUNCS[lossF]
        self.loss_derivative = LOSS_DERIVATES[lossF]
        hiddenActivationFunc = ACTIVATION_FUNCTIONS[hiddenAf]
        self.hiddenDerivative = ACTIVATION_DERIVATIVES[hiddenAf]
        self.outAF = outAf
        outputActivationFunc = ACTIVATION_FUNCTIONS[outAf]
        self.outputDerivative = ACTIVATION_DERIVATIVES[outAf]
        self.activations = [
            hiddenActivationFunc for _ in range(self.num_layers - 1)
        ]
        self.activations.append(outputActivationFunc)
        assert len(self.activations) == self.num_layers  # len(sizes)

    def feedforward(self, x):
        a = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        # (num_features, num_examples)
        for b, w, af in zip(self.biases, self.weights, self.activations):
            z = self.xp.dot(w, a) + b
            cp.cuda.Stream.null.synchronize()
            zs.append(z)
            a = af(z)
            activations.append(a)
        return zs, activations, a

    def train(
        self,
        trainX,
        trainY,
        epochs,
        mini_batch_size=1,
        eta=1e-3,
        testX=None,
        testY=None,
        save=False
    ):
        best_weights, best_biases, best_accuracy = [], [], 0

        n = len(trainX)
        costList = []
        accList = []
        if testX is None:
            n_test = n
            testX = trainX.copy().T
            testY = trainY.copy()
        else:
            n_test = len(testX)
            testX = testX.T
            # No need to keep this in hot vector encoded form
        if testY.shape[0] != testY.size:
            testY = testY.argmax(axis=1)
        testY = testY.reshape(1, -1)
        print("Starting training")
        for j in range(epochs):
            # random shuffling
            p = self.xp.random.permutation(n)
            epochCost = []
            trainX = trainX[p, :]
            trainY = trainY[p, :]

            mini_batches = [
                (
                    trainX[k : k + mini_batch_size, :],
                    trainY[k : k + mini_batch_size, :],
                )
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                miniBatchCost = self.update_mini_batch(mini_batch, eta)
                epochCost.append(miniBatchCost)
            if testX is not None:
                correct = self.get_accuracy(testX, testY)
                cp.cuda.Stream.null.synchronize()
                acc = correct * 100.0 / n_test
                cost = self.xp.array(epochCost).mean()
                accList.append(acc)
                costList.append(cost)
                print(
                    "Epoch {0}: {1} / {2} ({3:.05f}%)\tTest Loss: {4:.02f}".format(
                        j + 1, correct, n_test, float(acc), float(cost)
                    )
                )
                if save and acc > best_accuracy:
                    best_accuracy = acc
                    best_weights = self.weights.copy()
                    best_biases = self.biases.copy()
            else:
                cost = self.xp.array(epochCost).mean()
                costList.append(cost)
                print(
                    "Epoch {0}\tTrain Loss {1:.02f}".format(j + 1, float(cost))
                )
        if save:
            saveNpy([best_weights, best_biases])
        return costList, accList

    def update_mini_batch(self, mini_batch, eta):
        x, y = mini_batch
        m = x.shape[0] * 1.0
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
        old_weights = self.weights[:]
        old_biases = self.biases[:]
        self.weights = [w - (eta * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb) for b, nb in zip(self.biases, nabla_b)]

        cp.cuda.Stream.null.synchronize()
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
        # return (nabla_b, nabla_w, None)

    def get_accuracy(self, testX, testY):
        # testY should NOT be one hot encoded for this to work
        # The code at the start of training takes care of it if testY was onehot encoded
        # when passed into the train func
        _, _, y_hat = self.feedforward(testX)
        cp.cuda.Stream.null.synchronize()
        preds = y_hat.argmax(axis=0).reshape(1, -1)
        return (testY == preds).sum()
