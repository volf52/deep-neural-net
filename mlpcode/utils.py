import struct
from pathlib import Path

import cupy as cp
import numpy as np
import struct
from enum import Enum

# Data dir is just a folder named 'data' in the same directory where this file exists. It should contain
# mnist data (all four files, uncompressed)
DATA_DIR: Path = Path(__file__).parent / "data"
assert DATA_DIR.exists()

MNIST_DIR: Path = DATA_DIR / "mnist"
FASHION_MNIST_DIR: Path = DATA_DIR / "fashion-mnist"
MNIST_C_DIR: Path = DATA_DIR / "mnist-c"

MNIST_CLASSES = 10


class DATASETS(Enum):
    mnist = "mnist"
    fashion = "fashion-mnist"
    # mnistc = "mnist-c"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def oneHotEncoding(classes: int, y):
    xp = cp.get_array_module(y)
    return xp.eye(classes, dtype=xp.uint8)[y]


def loadFile(
    file_pth: Path, isTest: bool, useGpu=True,
):
    if useGpu:
        xp = cp
    else:
        xp = np

    with file_pth.open("rb") as f:
        if isTest:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch for testing data. {} != 2049".format(
                        magic
                    )
                )
        else:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch for testing data. {} != 2051".format(
                        magic
                    )
                )
        data = xp.fromfile(f, dtype=xp.uint8)

    return data


def loadX(file_pth: Path, num_instances: int, num_features: int, useGpu=True):
    X = (
        loadFile(file_pth, False, useGpu)
        .astype(np.float32)
        .reshape(num_instances, num_features)
    )
    return X / 255.0


def loadY(file_pth: Path, num_instances: int, useGpu=True, encoded=True):
    y = loadFile(file_pth, True, useGpu)
    if encoded:
        y = oneHotEncoding(MNIST_CLASSES, y)
    else:
        y = y.reshape(-1, 1)

    return y


def loadTesting(dataDir: Path, useGpu=True, encoded=True):
    xPth: Path = dataDir / "t10k-images-idx3-ubyte"
    yPth: Path = dataDir / "t10k-labels-idx1-ubyte"
    assert xPth.exists()
    assert yPth.exists()
    instances = 10000
    X = loadX(xPth, instances, 784, useGpu)
    y = loadY(yPth, instances, useGpu, encoded)
    return X, y


def loadTraining(dataDir: Path, useGpu=True, encoded=True):
    xPth: Path = dataDir / "train-images-idx3-ubyte"
    yPth: Path = dataDir / "train-labels-idx1-ubyte"
    assert xPth.exists()
    assert yPth.exists()
    instances = 60000
    X = loadX(xPth, instances, 784, useGpu)
    y = loadY(yPth, instances, useGpu, encoded)
    return X, y


def loadMnist(useGpu=True, encoded=True):
    trainX, trainY = loadTraining(MNIST_DIR, useGpu, encoded)
    testX, testY = loadTesting(MNIST_DIR, useGpu, encoded)
    return trainX, trainY, testX, testY


def loadFashionMnist(useGpu=True, encoded=True):
    trainX, trainY = loadTraining(FASHION_MNIST_DIR, useGpu, encoded)
    testX, testY = loadTesting(FASHION_MNIST_DIR, useGpu, encoded)
    return trainX, trainY, testX, testY


LOADING_FUNCS = {DATASETS.mnist: loadMnist, DATASETS.fashion: loadFashionMnist}


def loadDataset(dataset: DATASETS, useGpu=True, encoded=True):
    return LOADING_FUNCS[dataset](useGpu, encoded)


if __name__ == "__main__":
    trainX, trainY, testX, testY = loadDataset(DATASETS.fashion)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
