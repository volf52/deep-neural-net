import struct
from pathlib import Path

import cupy as cp
import numpy as np
import struct
import json
from enum import Enum

# CONFIGURATION
CONFIG_FILE = Path("./nn.config.json")
if CONFIG_FILE.exists():
    with CONFIG_FILE.open("r") as f:
        config = json.load()
else:
    prnt = Path(__file__).parent
    config = {"DATADIR": prnt / "data", "MODELDIR": prnt / "models"}
    if not config["MODELDIR"].exists():
        config["MODELDIR"].mkdir()

# Data dir is just a folder named 'data' in the same directory where this file exists.
# It should contain the required datasets on which the models are to be trained/tested
DATADIR: Path = config["DATADIR"]
assert DATADIR.exists()

MODELDIR: Path = config["MODELDIR"]

MNISTDIR: Path = DATADIR / "mnist"
FASHIONMNISTDIR: Path = DATADIR / "fashion-mnist"
MNISTCDIR: Path = DATADIR / "mnist-c"

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
                    "Magic number mismatch for testing data. {} != 2049".format(magic)
                )
        else:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch for testing data. {} != 2051".format(magic)
                )
        data = xp.fromfile(f, dtype=xp.uint8)

    return data


def loadNpyFile(file_pth: Path, isTest, useGpu=True):
    # isTest won't be used. It's just there for consistency (being able to use HOF)
    if useGpu:
        xp = cp
    else:
        xp = np
    with file_pth.open("rb") as f:
        data = xp.load(f)
    return data


def loadX(
    file_pth: Path, loadFunc, num_instances: int, num_features: int, useGpu=True,
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
    dirPth: Path = MNISTCDIR / subCat

    assert dirPth.exists()
    assert dirPth.is_dir()

    trainX, trainY = loadNpyTraining(dirPth, useGpu, encoded)
    testX, testY = loadNpyTesting(dirPth, useGpu, encoded)
    return trainX, trainY, testX, testY


def loadMnist(useGpu=True, encoded=True):
    trainX, trainY = loadTraining(MNISTDIR, useGpu, encoded)
    testX, testY = loadTesting(MNISTDIR, useGpu, encoded)
    return trainX, trainY, testX, testY


def loadFashionMnist(useGpu=True, encoded=True):
    trainX, trainY = loadTraining(FASHIONMNISTDIR, useGpu, encoded)
    testX, testY = loadTesting(FASHIONMNISTDIR, useGpu, encoded)
    return trainX, trainY, testX, testY


LOADING_FUNCS = {DATASETS.mnist: loadMnist, DATASETS.fashion: loadFashionMnist}


def loadDataset(dataset: DATASETS, useGpu=True, encoded=True):
    if str(dataset).startswith("mnist_c"):
        return loadMnistC(dataset)
    else:
        return LOADING_FUNCS[dataset](useGpu, encoded)


if __name__ == "__main__":
    # trainX, trainY, testX, testY = loadDataset(DATASETS.fashion)
    trainX, trainY, testX, testY = loadDataset(DATASETS.mnistc_identity)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
