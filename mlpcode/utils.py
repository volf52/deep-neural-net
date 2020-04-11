import struct
from pathlib import Path

import cupy as cp
import numpy as np
from mnist import MNIST

# Data dir is just a folder named 'data' in the same directory where this file exists. It should contain
# mnist data (all four files, uncompressed)
DATA_DIR: Path = Path(__file__).parent / "data"
assert DATA_DIR.exists()

mnist_data = MNIST(DATA_DIR)

MNIST_CLASSES = 10


def oneHotEncoding(classes: int, y):
    xp = cp.get_array_module(y)
    return xp.eye(classes)[y.astype("int32")]


def loadData(load_func, useGpu=True, encoded=True, classes=MNIST_CLASSES):
    if useGpu:
        xp = cp
    else:
        xp = np

    data, labels = load_func()

    X = xp.array(data, dtype=xp.float64) / 255.0
    y = xp.array(labels, dtype=xp.uint8)

    if encoded:
        y = oneHotEncoding(classes, y)

    return X, y


def read_test(useGpu=True, encoded=True, classes=MNIST_CLASSES):
    return loadData(mnist_data.load_testing, useGpu, encoded, classes)


def read_train(useGpu=True, encoded=True, classes=MNIST_CLASSES):
    return loadData(mnist_data.load_training, useGpu)


if __name__ == "__main__":
    # Ensure that you change the '.' before idx to '-'
    # example train-labels.idx1-ubyte to train-labels-idx1-ubyte
    trainX, trainY = read_train()
    testX, testY = read_test()

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
