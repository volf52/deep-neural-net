import numpy as np
import cupy as cp

from mlpcode.activation import ActivationFuncs as af
from mlpcode.activation import ACTIVATION_FUNCTIONS, ACTIVATION_DERIVATIVES


class Network(object):
    def __init__(
        self,
        sizes,
        useGpu=False,
        hiddenAf: af = af.sigmoid,
        outAf: af = af.sigmoid,
    ):
        assert hiddenAf in ACTIVATION_FUNCTIONS
        assert outAf in ACTIVATION_FUNCTIONS
        self.num_layers = len(sizes)
        if useGpu:
            self.xp = cp
        else:
            self.xp = np
        self.sizes = sizes
        self.biases = [self.xp.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            self.xp.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]
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
        mini_batch_size,
        eta,
        testX=None,
        testY=None,
    ):
        if testX is not None:
            n_test = len(testX)
            testX = testX.T
            testY = testY.T
        n = len(trainX)
        for j in range(epochs):
            # random shuffling
            p = self.xp.random.permutation(n)
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
                self.update_mini_batch(mini_batch, eta)
            if testX is not None:
                correct = self.get_accuracy(testX, testY)
                acc = correct * 100.0 / n_test
                print(
                    "Epoch {0}: {1} / {2} ({3}%)".format(
                        j, correct, n_test, acc
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [self.xp.zeros(b.shape) for b in self.biases]
        nabla_w = [self.xp.zeros(w.shape) for w in self.weights]
        x, y = mini_batch
        m = x.shape[0] * 1.0
        delta_nabla_b, delta_nabla_w = self.backprop(x.T, y.T)
        nabla_b = [nb.mean(axis=1, keepdims=True) for nb in delta_nabla_b]
        nabla_w = [(nw / m) for nw in delta_nabla_w]
        self.weights = [w - (eta * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [None for b in self.biases]
        nabla_w = [None for w in self.weights]

        # forward pass
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w, af in zip(self.biases, self.weights, self.activations):
            z = self.xp.dot(w, activation) + b
            zs.append(z)
            activation = af(z)
            activations.append(activation)

        # backward pass
        outDeriv = self.outputDerivative
        delta = self.cost_derivative(activations[-1], y) * outDeriv(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = self.xp.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.hiddenDerivative(z)
            delta = self.xp.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = self.xp.dot(delta, activations[-l - 1].T)
        return (nabla_b, nabla_w)

    def get_accuracy(self, testX, testY):
        y_hat = self.feedforward(testX)
        pred = self.xp.argmax(y_hat, axis=0)
        return (testY == pred).sum()

    def cost_derivative(self, y_hat, y):
        return y_hat - y


# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0 / (1.0 + np.exp(-z))


# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    from mlpcode.utils import read_train, read_test

    useGpu = True
    X_train, y_train = read_train(reshape=True, useGpu=useGpu)
    X_test, y_test = read_test(reshape=True, useGpu=useGpu)
    layers = [784, 64, 10]
    out = 2
    epochs = 100
    m = 60000

    nn = Network(layers, useGpu=useGpu)
    nn.train(X_train, y_train, epochs, 600, 1e-3, X_test, y_test)
