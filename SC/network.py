from multiprocessing import Pool, cpu_count

import numpy as np

from SC.math import dot
from SC.utils import mat_bpe_decode, mat2SC, loadDataset
from mlpcode.network import Network
from SC.utils import DATASETS
from mlpcode.activation import ACTIVATION_FUNCTIONS


class SCNetwork:
    def __init__(
        self, network: Network, hiddenAf, outAf, precision, binarized=False
    ):
        self.__hiddenAf = ACTIVATION_FUNCTIONS[hiddenAf]
        self.__outAf = ACTIVATION_FUNCTIONS[outAf]
        weights = network.weights.copy()
        biases = network.biases.copy()
        self.num_layers = network.num_layers

        if binarized:
            weights = [Network.binarize(weights[i]) for i in range(self.num_layers)]
            biases = [Network.binarize(biases[i]) for i in range(self.num_layers)]

        for i in range(self.num_layers):
            weights[i] = np.append(weights[i], biases[i], axis=1)

        self.weights = [mat2SC(weights[i], precision=precision) for i in range(self.num_layers)]
        self.precision = precision
        self.activations = [self.__hiddenAf for i in range(self.num_layers - 1)]
        self.activations.append(self.__outAf)

    def forwardpass(self, x):
        a = x
        for w, af in zip(self.weights, self.activations):
            a = np.append(
                a,
                mat2SC(np.ones((1, a.shape[1])), precision=self.precision),
                axis=0
            )
            z = dot(w, a, conc=10)
            a = mat2SC(
                af(mat_bpe_decode(z, precision=self.precision, conc=10)),
                precision=self.precision
            )
        return a

    def get_accuracy(self, x, y):
        preds = self.forwardpass(x)
        preds = mat_bpe_decode(preds, precision=self.precision)
        preds = preds.argmax(axis=0).reshape(1, -1)
        return (preds == y).sum()

    def testDataset(self, dataset: DATASETS, num_instances=10000, parallel=False):
        if parallel:
            return self.__testDatasetParallel(dataset, num_instances)

        batch_size = 50
        overall_correct = 0
        for i in range(0, num_instances, 1000):
            x, y, detY = loadDataset(dataset, precision=self.precision, idx=i)
            for j in range(0, 1000, batch_size):
                overall_correct += self.get_accuracy(x[:, i:i + batch_size], detY[:, i:i + batch_size])

        return overall_correct

    def __testDatasetParallel(self, dataset: DATASETS, num_instances=10000):
        batch_size = 50
        # Needs optimization, currently doesn't get use of > 20 processes
        processCount = cpu_count()
        if processCount > 20:
            processCount = 20

        pool = Pool(processes=processCount)
        overall_correct = 0
        for i in range(0, num_instances, 1000):
            x, y, detY = loadDataset(dataset, precision=self.precision, idx=i)
            input_args = [(x[:, j: j + batch_size], detY[:, j:j + batch_size]) for j in range(0, 1000, batch_size)]
            output = pool.starmap(self.get_accuracy, input_args)
            overall_correct += sum(output)

        pool.close()
        return overall_correct