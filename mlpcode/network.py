import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf


class Network(object):
    def __init__(
        self,
        sizes,
        useGpu=False,
        hiddenAf: af = af.sigmoid,
        outAf: af = af.sigmoid,
        lossF: lf = lf.mse,
    ):
        assert hiddenAf in ACTIVATION_FUNCTIONS
        assert outAf in ACTIVATION_FUNCTIONS
        assert lossF in LOSS_FUNCS
        self.num_layers = len(sizes)
        if useGpu:
            self.xp = cp
        else:
            self.xp = np
        self.sizes = sizes
        self.biases = [self.xp.random.randn(y, 1) for y in sizes[1:]]
        # Xavier init
        self.weights = [
            (self.xp.random.randn(l, l_minus_1) * self.xp.sqrt(1 / l_minus_1))
            for l_minus_1, l in zip(sizes[:-1], sizes[1:])
        ]
        cp.cuda.Stream.null.synchronize()
        self.loss = LOSS_FUNCS[lossF]
        self.loss_derivative = LOSS_DERIVATES[lossF]
        hiddenActivationFunc = ACTIVATION_FUNCTIONS[hiddenAf]
        self.hiddenDerivative = ACTIVATION_DERIVATIVES[hiddenAf]
        outputActivationFunc = ACTIVATION_FUNCTIONS[outAf]
        self.outputDerivative = ACTIVATION_DERIVATIVES[outAf]
        self.activations = [
            hiddenActivationFunc for _ in range(self.num_layers - 1)
        ]
        self.activations.append(outputActivationFunc)
        assert len(self.activations) == self.num_layers  # len(sizes)

    def feedforward(self, a):
        for b, w, af in zip(self.biases, self.weights, self.activations):
            a = af(self.xp.dot(w, a) + b)
        return a

    def train(
        self,
        trainX,
        trainY,
        epochs,
        mini_batch_size=1,
        eta=1e-3,
        testX=None,
        testY=None,
    ):
        if testX is not None:
            n_test = len(testX)
            testX = testX.T
            testY = testY.T
        n = len(trainX)
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
                # epochCost.append(miniBatchCost)
            if testX is not None:
                correct = self.get_accuracy(testX, testY)
                acc = correct * 100.0 / n_test
                # cost = self.xp.array(epochCost).mean()
                cost = "not calculating for now"
                print(
                    "Epoch {0}: {1} / {2} ({3}%)\tLoss: {4}".format(
                        j, correct, n_test, acc, cost
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        x, y = mini_batch
        m = x.shape[0] * 1.0
        delta_nabla_b, delta_nabla_w, cost = self.backprop(x.T, y.T)
        cp.cuda.Stream.null.synchronize()
        nabla_b = [nb.mean(axis=1, keepdims=True) for nb in delta_nabla_b]
        nabla_w = [(nw / m) for nw in delta_nabla_w]
        cp.cuda.Stream.null.synchronize()
        self.weights = [w - (eta * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb) for b, nb in zip(self.biases, nabla_b)]
        cp.cuda.Stream.null.synchronize()
        return cost

    def backprop(self, x, y):
        nabla_b = [None for b in self.biases]
        nabla_w = [None for w in self.weights]

        # forward pass
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w, af in zip(self.biases, self.weights, self.activations):
            preActiv = self.xp.dot(w, activation) + b
            cp.cuda.Stream.null.synchronize()
            zs.append(preActiv)
            activation = af(preActiv)
            activations.append(activation)

        # Mean cost of whole batch
        # cost = self.loss(activation, y).mean()
        # backward pass
        dLdA = self.loss_derivative(activation, y)  # expected shape: k * m
        cp.cuda.Stream.null.synchronize()
        dAdZ = self.outputDerivative(dLdA, zs[-1])
        delta = dLdA * dAdZ
        nabla_b[-1] = delta
        nabla_w[-1] = self.xp.dot(delta, activations[-2].T)
        cp.cuda.Stream.null.synchronize()

        for l in range(2, self.num_layers):
            z = zs[-l]
            dAprev = self.xp.dot(self.weights[-l + 1].T, delta)
            cp.cuda.Stream.null.synchronize()
            delta = dAprev * self.hiddenDerivative(dAprev, z)
            cp.cuda.Stream.null.synchronize()
            nabla_b[-l] = delta
            nabla_w[-l] = self.xp.dot(delta, activations[-l - 1].T)
            cp.cuda.Stream.null.synchronize()
        # return (nabla_b, nabla_w, cost)
        return (nabla_b, nabla_w, None)

    def get_accuracy(self, testX, testY):
        y_hat = self.feedforward(testX)
        cp.cuda.Stream.null.synchronize()
        pred = y_hat.argmax(axis=0, keepdims=True)
        return (testY == pred).sum()
