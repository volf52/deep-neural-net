import random

# Third-party libraries
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        trainX,
        trainY,
        epochs,
        mini_batch_size,
        eta,
        testX=None,
        testY=None,
    ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if testX is not None:
            n_test = len(testX)
            testX = testX.T
            testY = testY.T
        n = len(trainX)
        # trainX = trainX.T
        # trainY = trainY.T
        for j in range(epochs):
            # random shuffling
            p = np.random.permutation(n)
            trainX = trainX[p, :]
            trainY = trainY[p, :]

            mini_batches = [
                (
                    # trainX[:, k : k + mini_batch_size],
                    # trainY[:, k : k + mini_batch_size],
                    trainX[k : k + mini_batch_size, :],
                    trainY[k : k + mini_batch_size, :],
                )
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if testX is not None:
                correct = self.evaluate(testX, testY)
                acc = correct * 100.0 / n_test
                print(
                    "Epoch {0}: {1} / {2} ({3}%)".format(
                        j, correct, n_test, acc
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # Next two lines not really required
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # for x, y in zip(*mini_batch):
        x, y = mini_batch
        m = x.shape[0] * 1.0
        # delta_nabla_b, delta_nabla_w = self.backprop(
        #     x.reshape(-1, 1), y.reshape(-1, 1)
        # )
        # nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        delta_nabla_b, delta_nabla_w = self.backprop(x.T, y.T)
        nabla_b = [nb.mean(axis=1, keepdims=True) for nb in delta_nabla_b]
        nabla_w = [(nw / m) for nw in delta_nabla_w]
        # nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # w_0 = np.copy(self.weights[0])
        # w_1 = np.copy(self.weights[1])
        self.weights = [w - (eta * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb) for b, nb in zip(self.biases, nabla_b)]
        # lf0 = np.array_equal(w_0, self.weights[0])
        # lf1 = np.array_equal(w_1, self.weights[1])

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [None for b in self.biases]
        nabla_w = [None for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(
            zs[-1]
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
            # nabla_w[-l] = np.dot(delta, activations[-l].T)
        return (nabla_b, nabla_w)

    def evaluate(self, testX, testY):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        y_hat = self.feedforward(testX)
        pred = np.argmax(y_hat, axis=0)
        # test_results = [
        #     (np.argmax(self.feedforward(x)), y) for (x, y) in zip(testX, testY)
        # ]
        return (testY == pred).sum()

    def cost_derivative(self, y_hat, y):
        return y_hat - y


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    from mlpcode.utils import read_train, read_tenk

    X_train, y_train = read_train(reshape=True, useGpu=False)
    X_test, y_test = read_tenk(reshape=True, useGpu=False)
    layers = [784, 64, 10]
    out = 2
    epochs = 100
    m = 60000

    nn = Network(layers)
    # print(nn.weights)
    nn.SGD(X_train, y_train, epochs, 600, 1e-3, X_test, y_test)
