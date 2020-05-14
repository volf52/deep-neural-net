import struct
from pathlib import Path

import cupy as cp
import numpy as np
import struct
import json
from enum import Enum

prnt = Path(__file__).parent

# CONFIGURATION
CONFIG_FILE = prnt / "nn.config.json"
if CONFIG_FILE.exists():
    print(f"Reading config from {CONFIG_FILE}")
    with CONFIG_FILE.open("r") as f:
        tmp = json.load(f)
        config = {k: Path(tmp[k]) for k in tmp.keys()}
else:
    config = {"DATADIR": prnt / "data", "MODELDIR": prnt / "models"}
    if not config["MODELDIR"].exists():
        config["MODELDIR"].mkdir()
    with CONFIG_FILE.open("w") as f:
        json.dump({k: str(config[k]) for k in config.keys()}, f)
    print(f"Wrote config to {CONFIG_FILE}")


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
    cifar10 = "cifar-10"
    # affnist = 'affNIST'

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
    if y.ndim > 1:
        y = xp.squeeze(y)
    assert y.ndim == 1
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
    # with file_pth.open("rb") as f:
    #     data = xp.load(f)
    data = xp.load(file_pth)
    cp.cuda.Stream.null.synchronize()
    return data


def loadX(
    file_pth: Path, loadFunc, num_instances: int, num_features: int, useGpu=True,
):
    X = (
        loadFunc(file_pth, False, useGpu).reshape(num_instances, num_features) / 255.0
    ).astype(np.float32)
    # using inplace operator to not waste memory on copying and operating on a copy
    # moved division with 255 above to avoid recasting problems
    # X /= 255.0
    cp.cuda.Stream.null.synchronize()
    return X


def loadY(file_pth: Path, loadFunc, useGpu=True, encoded=True):
    y = loadFunc(file_pth, True, useGpu).astype(np.uint8)
    if encoded:
        y = oneHotEncoding(MNIST_CLASSES, y)
    else:
        y = y.reshape(-1, 1)

    cp.cuda.Stream.null.synchronize()
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


def loadNpyTesting(
    dataDir: Path, instances: int, inputNeurons: int, useGpu=True, encoded=True
):
    xPth = dataDir / "test_images.npy"
    yPth = dataDir / "test_labels.npy"
    assert xPth.exists()
    assert yPth.exists()
    X = loadX(xPth, loadNpyFile, instances, inputNeurons, useGpu)
    y = loadY(yPth, loadNpyFile, useGpu, encoded)
    return X, y


def loadNpyTraining(
    dataDir: Path, instances: int, inputNeurons: int, useGpu=True, encoded=True
):
    xPth = dataDir / "train_images.npy"
    yPth = dataDir / "train_labels.npy"
    assert xPth.exists()
    assert yPth.exists()
    X = loadX(xPth, loadNpyFile, instances, inputNeurons, useGpu)
    y = loadY(yPth, loadNpyFile, useGpu, encoded)
    return X, y


def loadMnistC(category: DATASETS, useGpu=True, encoded=True):
    val: str = str(category)
    assert val.startswith("mnist_c")
    subCat = val.split("-")[-1]
    dirPth: Path = MNISTCDIR / subCat

    assert dirPth.exists()
    assert dirPth.is_dir()

    trainInstances = 60000
    testInstances = 10000
    inputNeurons = 784

    trainX, trainY = loadNpyTraining(
        dirPth, trainInstances, inputNeurons, useGpu=useGpu, encoded=encoded
    )
    testX, testY = loadNpyTesting(
        dirPth, testInstances, inputNeurons, useGpu=useGpu, encoded=encoded
    )
    return trainX, trainY, testX, testY


def loadMnist(useGpu=True, encoded=True):
    trainX, trainY = loadTraining(MNISTDIR, useGpu, encoded)
    testX, testY = loadTesting(MNISTDIR, useGpu, encoded)
    return trainX, trainY, testX, testY


def loadFashionMnist(useGpu=True, encoded=True):
    trainX, trainY = loadTraining(FASHIONMNISTDIR, useGpu, encoded)
    testX, testY = loadTesting(FASHIONMNISTDIR, useGpu, encoded)
    return trainX, trainY, testX, testY


def loadAffNist(useGpu=True, encoded=True):
    root = DATADIR / "affnist"
    prefix = "affNIST"
    train_instances = 1600000
    test_validation_instances = 320000
    features = 1600
    trainX = loadX(
        root / f"{prefix}_trainX.npy", loadNpyFile, train_instances, features, useGpu,
    )
    trainY = loadY(root / f"{prefix}_trainY.npy", loadNpyFile, useGpu, encoded)
    testX = loadX(
        root / f"{prefix}_testX.npy",
        loadNpyFile,
        test_validation_instances,
        features,
        useGpu,
    )
    testY = loadY(root / f"{prefix}_testY.npy", loadNpyFile, useGpu, encoded)
    return trainX, trainY, testX, testY


def loadCifar10(useGpu=True, encoded=True):
    root = DATADIR / "cifar-10"
    prefix = "cifar-10_greyscale"
    train_instances = 50000
    test_validation_instances = 10000
    features = 1024
    trainX = loadX(
        root / f"{prefix}_trainX.npy", loadNpyFile, train_instances, features, useGpu,
    )
    trainY = loadY(root / f"{prefix}_trainY.npy", loadNpyFile, useGpu, encoded)
    testX = loadX(
        root / f"{prefix}_testX.npy",
        loadNpyFile,
        test_validation_instances,
        features,
        useGpu,
    )
    testY = loadY(root / f"{prefix}_testY.npy", loadNpyFile, useGpu, encoded)
    return trainX, trainY, testX, testY


LOADING_FUNCS = {
    DATASETS.mnist: loadMnist,
    DATASETS.fashion: loadFashionMnist,
    DATASETS.cifar10: loadCifar10,
}


def loadDataset(dataset: DATASETS, useGpu=True, encoded=True):
    if str(dataset).startswith("mnist_c"):
        return loadMnistC(dataset)
    else:
        return LOADING_FUNCS[dataset](useGpu, encoded)


if __name__ == "__main__":
    # trainX, trainY, testX, testY = loadDataset(DATASETS.cifar10)
    # trainX, trainY, testX, testY = loadAffNist(useGpu=False)
    trainX, trainY, testX, testY = loadCifar10()

    print((trainX.shape, trainX.dtype))
    print((trainY.shape, trainY.dtype))
    print((testX.shape, testX.dtype))
    print((testY.shape, testY.dtype))
