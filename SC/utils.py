import os
from numbers import Number
from pathlib import Path

import numpy as np

import mlpcode.utils

DATASETS = mlpcode.utils.DATASETS

DATA_DIR: Path = Path(__file__).parent / "data"
assert DATA_DIR.exists()

MNIST_DIR: Path = DATA_DIR / "mnist"
FASHION_MNIST_DIR: Path = DATA_DIR / "fashion-mnist"
MNIST_C_DIR: Path = DATA_DIR / "mnist-c"


# To quantize a float or an array of floats
# returns : float or an array of floats
def quantize(x, min=-1, max=1, precision=16):
    if isinstance(x, np.ndarray):
        x[x <= min] = min
        x[x >= max] = max
        q = (max - min) / precision
        return q * np.round(x / q)

    if x >= max:
        return max
    elif x <= min:
        return min
    else:
        q = (max - min) / precision
        return q * np.round(x / q)


# Calculates the Bipolar encoding probability of a parameter x
def bpe_encode(x):
    # x is expected to be quantized
    return (x + 1) / 2

# Converts bipolar encoded bit-stream to decimal
def bpe_decode(x: np.ndarray, precision, conc=1):
    result = 0
    i = 0
    if conc > 1:
        tmp = (2 * np.count_nonzero(x[i:precision + i]) - precision) / precision
        while conc != 0 and tmp != 0:
            result += tmp
            i += precision
            conc -= 1
            tmp = (2 * np.count_nonzero(x[i:precision + i]) - precision) / precision

    else:
        result = (2 * np.count_nonzero(x) - precision) / precision

    return result


# vectorized version of bpe_decode()
def vect_bpe_decode(x: np.ndarray, precision, conc=1):
    # x is a 2D numpy array holding bit-streams
    n = len(x)
    result = np.ndarray(n, np.float)
    for i in range(n):
        result[i] = bpe_decode(x[i], precision, conc=conc)
    return result


def mat_bpe_decode(x: np.ndarray, precision, conc=1):
    # x is a 3D numpy array holding bit-streams
    # returns 2D float numpy array
    result = np.ndarray((x.shape[0], x.shape[1]), dtype=np.float)
    for i in range(len(x)):
        result[i] = vect_bpe_decode(x[i], precision, conc=conc)
    return result

# Generates a random bit stream given length and number of ones
def random_stream(length, n_ones) -> np.ndarray:
    stream = np.ndarray(length, dtype=np.bool)
    stream[0:n_ones] = 1
    stream[n_ones:] = 0
    np.random.shuffle(stream)
    '''
    # a slower way
    stream[0:]=0
    indices = np.random.choice(np.arange(start=0, stop=length,dtype = np.bool), n_ones, replace=False)
    stream.put(indices,1)
    '''
    return stream


# Converts a decimal to bipolar encoded stochastic bit-stream
def SCNumber(x: Number, min=-1, max=1, precision=16) -> np.ndarray:
    x = quantize(x, min, max, precision)
    prob = bpe_encode(x)
    n_ones = int(prob * precision)
    return random_stream(precision, n_ones)


# vectorized version of SCNumber
def vect2SC(x: np.ndarray, min=-1, max=1, precision=16) -> np.ndarray:
    # x : 1D numpy float array
    # returns 2D numpy bol array that contains bit-streams
    n = len(x)
    result = np.ndarray((n, precision), dtype=np.bool)
    x = quantize(x, min, max, precision)
    prob: np.ndarray = bpe_encode(x)
    n_ones = (prob * precision).astype(int)
    for i in range(n):
        result[i] = random_stream(precision, n_ones[i])
    return result


def mat2SC(x: np.ndarray, min=-1, max=1, precision=16) -> np.ndarray:
    # x : 2D numpy float array
    # returns 3D numpy bool array that contains bit-streams
    n = len(x)
    result = np.ndarray((x.shape[0], x.shape[1], precision), dtype=np.bool)
    for i in range(n):
        result[i] = vect2SC(x[i], min, max, precision)
    return result


# noinspection PyTypeChecker
# Converts a dataset to stochastic bit-stream and writes it on the disk
# it's written in 'data' folder in the same directory where the file exists
def dataset2SC(dataset: DATASETS, precision):
    val = str(dataset)
    num_instances = 10000

    # todo : add cifar-10 & affNist

    if val.startswith('mnist_c'):
        write_dir = MNIST_C_DIR / val.split("-")[-1]
    else:
        if val.startswith('fashion'):
            write_dir = FASHION_MNIST_DIR
        else:
            write_dir = MNIST_DIR

    if not write_dir.is_dir():
        os.mkdir(write_dir)
    write_dir = write_dir / str(precision)

    if not write_dir.is_dir():
        os.mkdir(write_dir)
        os.mkdir(write_dir / 'labels')
        os.mkdir(write_dir / 'images')

    _, _, testX, testY = mlpcode.utils.loadDataset(dataset, useGpu=False)   # Redundant operation

    # taking care of oneHot-encoding
    testY = testY.argmax(axis=1)
    testY = testY.reshape(1, -1)

    # (num_instances, num_features) --> (num_features, num_instances)
    # each column represents an instance
    testX = testX.T

    for i in range(0, num_instances, 1000):
        Y = mat2SC(testY[:, i:i + 1000], precision=precision)
        X = mat2SC(testX[:, i:i + 1000], precision=precision)
        np.save(write_dir / 'labels' / str(i), Y)
        np.save(write_dir / 'images' / str(i), X)

        # Temporary
        np.save(write_dir / 'labels' / (str(i) + '_deterministic'), testY[:, i:i + 1000])


# noinspection PyTypeChecker
def loadDataset(dataset: DATASETS, precision, idx):
    dataset = str(dataset)
    if dataset.startswith('fashion'):
        load_dir = FASHION_MNIST_DIR

    elif dataset.startswith("mnist_c"):
        load_dir = MNIST_C_DIR / dataset.split("-")[-1]

    else:
        load_dir = MNIST_DIR

    load_dir = load_dir / str(precision)

    x = np.load(load_dir / 'images' / (str(idx) + '.npy'))
    y = np.load(load_dir / 'labels' / (str(idx) + '.npy'))
    deterministic_y = np.load(load_dir / 'labels' / (str(idx) + '_deterministic.npy'))

    return x, y, deterministic_y