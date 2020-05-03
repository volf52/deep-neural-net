import struct
from pathlib import Path

import cupy as cp
import numpy as np
import struct
from enum import Enum
import os

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
    mnistc_brightness = "mnist_c-brightness"
    mnistc_canny_edges = "mnist_c-canny_edges"
    mnistc_dotted_line = "mnist_c-dotted_line"
    mnistc_fog = "mnist_c-fog"
    mnistc_glass_blur = "mnist_c-glass_blur"
    mnistc_identity = "mnist_c-identity"
    mnistc_impulse_noise = "mnist_c-impulse_noise"
    mnistc_motion_blur = "mnist_c-motion_blur"
    mnistc_rotate = "mnist_c-rotate"
    mnistc_scale = "mnist_c-scale"
    mnistc_shear = "mnist_c-shear"
    mnistc_shot_noise = "mnist_c-shot_noise"
    mnistc_spatter = "mnist_c-spatter"
    mnistc_stripe = "mnist_c-stripe"
    mnistc_translate = "mnist_c-translate"
    mnistc_zigzag = "mnist_c-zigzag"

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


def loadIdxFile(
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


def loadNpyFile(file_pth: Path, isTest: bool, useGpu=True):
    if useGpu:
        xp = cp
    else:
        xp = np
    with file_pth.open("rb") as f:
        data = xp.load(f)
    return data


def loadX(
    file_pth: Path,
    loadFunc,
    num_instances: int,
    num_features: int,
    useGpu=True,
):
    X = (
        loadFunc(file_pth, False, useGpu)
        .astype(np.float32)
        .reshape(num_instances, num_features)
    )
    return X / 255.0


def loadY(file_pth: Path, loadFunc, useGpu=True, encoded=True):
    y = loadFunc(file_pth, True, useGpu)
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
    X = loadX(xPth, loadIdxFile, instances, 784, useGpu)
    y = loadY(yPth, loadIdxFile, useGpu, encoded)
    return X, y


def loadTraining(dataDir: Path, useGpu=True, encoded=True):
    xPth: Path = dataDir / "train-images-idx3-ubyte"
    yPth: Path = dataDir / "train-labels-idx1-ubyte"
    assert xPth.exists()
    assert yPth.exists()
    instances = 60000
    X = loadX(xPth, loadIdxFile, instances, 784, useGpu)
    y = loadY(yPth, loadIdxFile, useGpu, encoded)
    return X, y


def loadNpyTesting(dataDir: Path, useGpu=True, encoded=True):
    xPth = dataDir / "test_images.npy"
    yPth = dataDir / "test_labels.npy"
    assert xPth.exists()
    assert yPth.exists()
    instances = 10000
    X = loadX(xPth, loadNpyFile, instances, 784, useGpu)
    y = loadY(yPth, loadNpyFile, useGpu, encoded)
    return X, y


def loadNpyTraining(dataDir: Path, useGpu=True, encoded=True):
    xPth = dataDir / "train_images.npy"
    yPth = dataDir / "train_labels.npy"
    assert xPth.exists()
    assert yPth.exists()
    instances = 60000
    X = loadX(xPth, loadNpyFile, instances, 784, useGpu)
    y = loadY(yPth, loadNpyFile, useGpu, encoded)
    return X, y


def loadMnistC(category: DATASETS, useGpu=True, encoded=True):
    val: str = str(category)
    assert val.startswith("mnist_c")
    subCat = val.split("-")[-1]
    dirPth: Path = MNIST_C_DIR / subCat

    assert dirPth.exists()
    assert dirPth.is_dir()

    trainX, trainY = loadNpyTraining(dirPth, useGpu, encoded)
    testX, testY = loadNpyTesting(dirPth, useGpu, encoded)
    return trainX, trainY, testX, testY


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
    if str(dataset).startswith("mnist_c"):
        return loadMnistC(dataset)
    else:
        return LOADING_FUNCS[dataset](useGpu, encoded)

# lst = [weights, biases]
def saveNpy(lst : list, path = os.getcwd()):
    xp = cp.get_array_module(lst[0])
    if not os.path.isdir(path + '\\weights'):
        os.mkdir(path + '\\weights')

    if not os.path.isdir(path + '\\biases'):
        os.mkdir(path + '\\biases')
    for i, w in enumerate(lst[0]):
        xp.save(path+'\\weights\\'+str(i)+'.npy', w)

    for i, w in enumerate(lst[1]):
        xp.save(path+'\\biases\\'+str(i)+'.npy', w)


def loadWeightsBiasesNpy(xp, path=os.getcwd()):
    assert os.path.isdir(path+'\\weights')
    assert os.path.isdir(path+'\\biases')
    weights, biases = [], []
    for filename in os.listdir(path+'\\weights'):
        weights.append(xp.load(path+'\\weights\\'+filename))

    for filename in os.listdir(path+'\\biases'):
        biases.append(xp.load(path+'\\biases\\'+filename))

    return weights, biases


if __name__ == "__main__":
    # trainX, trainY, testX, testY = loadDataset(DATASETS.fashion)
    trainX, trainY, testX, testY = loadDataset(DATASETS.mnistc_identity)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
