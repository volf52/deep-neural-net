import json
import struct
from enum import Enum
from pathlib import Path
from typing import Tuple, Union

import cupy as cp
import numpy as np

# CONFIGURATION

prnt = Path(__file__).parent
CONFIG_FILE = prnt / "nn.config.json"

if CONFIG_FILE.exists():
    # print(f"Reading config from {CONFIG_FILE}")
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

TRAIN_TEST_DATA = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
XY_DATA = Tuple[np.ndarray, np.ndarray]


class DATASETS(Enum):
    mnist = "mnist"
    fashion = "fashion-mnist"
    cifar10 = "cifar-10"
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
    mnistc_contrast = "mnist_c-contrast"
    mnistc_defocus_blur = "mnist_c-defocus_blur"
    mnistc_elastic_transform = "mnist_c-elastic_transform"
    mnistc_frost = "mnist_c-frost"
    mnistc_gaussian_blur = "mnist_c-gaussian_blur"
    mnistc_gaussian_noise = "mnist_c-gaussian_noise"
    mnistc_inverse = "mnist_c-inverse"
    mnistc_jpeg_compression = "mnist_c-jpeg_compression"
    mnistc_line = "mnist_c-line"
    mnistc_pessimal_noise = "mnist_c-pessimal_noise"
    mnistc_pixelate = "mnist_c-pixelate"
    mnistc_quantize = "mnist_c-quantize"
    mnistc_saturate = "mnist_c-saturate"
    mnistc_snow = "mnist_c-snow"
    mnistc_speckle_noise = "mnist_c-speckle_noise"
    mnistc_zoom_blur = "mnist_c-zoom_blur"

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


# DATA UTILS


def oneHotEncode(classes: int, y: np.ndarray) -> np.ndarray:
    """
    One-hot encode a numpy array

    Parameters
    ----------
    classes
        Number of classes/categories (length of encoded vector)
    y
        Array to encode (usually the labels for training). Must be 1D or squeezable.

    Returns
    -------
    np.ndarray
        A 2D array of shape (y.shape[0], num_classes) and dtype np.int8

    """
    xp = cp.get_array_module(y)

    if y.ndim > 1:
        y = xp.squeeze(y)

    assert y.ndim == 1

    oneHotEncoded = xp.eye(classes, dtype=np.int8)[y]

    return oneHotEncoded


def quantize(x: np.ndarray, precision=32):
    """
    Inplace quantization. Doesn't return anything. Maps input range to evenly divided set of precision+1 elements

    Parameters
    ----------
    x
        The array to quantize
    precision
        The number of bits to create the intervals in the mapped set after quantization
    """

    xp = cp.get_array_module(x)
    intervals = 2 ** precision
    xp.multiply(x, intervals, out=x)
    xp.round_(x, out=x)
    xp.divide(x, intervals, out=x)


def split_train_valid(
    X: np.ndarray, y: np.ndarray, valSize: Union[int, float] = 0.2, shuffle=False
) -> TRAIN_TEST_DATA:
    """

    Parameters
    ----------
    X:
        np/cp array containing the data
    y
        np/cp array containing the labels
    valSize
        The size of validation data (float for percentage [0.1,0.99] and int for num_rows)
    shuffle
        Whether to shuffle the training data before taking the validation sample

    Returns
    -------
    TRAIN_TEST_DATA
        A 4-element tuple of numpy arrays (trainX, trainY, valX, valY) |
        trainX: Shape = (train_instances, features), dtype = np.float32 |
        trainY: Shape = (train_instances, num_classes or None), dtype = np.int8 |
        valX: Shape = (val_instances, features), dtype = np.float32 |
        valY: Shape = (val_instances, None), dtype = np.int8
    """
    n = X.shape[0]
    if isinstance(valSize, float):
        valSize = int(valSize * n)

    if shuffle:
        perms = np.random.permutation(n)
        X = X[perms]
        y = y[perms]

    return X[:-valSize], X[-valSize:], y[:-valSize], y[-valSize:]


# DATA I/O UTILS AND HELPER FUNCTIONS


def loadIdxFile(file_pth: Path, isTest: bool, useGpu=False) -> np.ndarray:
    """
    Load data from an IDX3 file

    Parameters
    ----------
    file_pth
        Path to the file containing the data
    isTest
        Whether the file contains testing data (to test the magic number of the file)
    useGpu
        Whether to use GPU or CPU as data device

    Returns
    -------
    np.ndarray
        A numpy array containing the data in file at file_pth
    """

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

        data = xp.fromfile(f, dtype=np.uint8)

    return data


def loadNpyFile(file_pth: Path, isLabelFile: bool, useGpu=False) -> np.ndarray:
    """
    Load data from an NPY file

    Parameters
    ----------
    file_pth
        Path to the file containing the data
    isLabelFile
        Whether the file contains labels (to test the magic number of the file)
    useGpu
        Whether to use GPU or CPU as data device

    Returns
    -------
    np.ndarray
        A numpy array containing the data in file at file_pth
    """

    if useGpu:
        xp = cp
    else:
        xp = np

    dt = np.int8 if isLabelFile else np.float32

    data = xp.load(file_pth).astype(dt)
    if useGpu:
        cp.cuda.Stream.null.synchronize()

    return data


def loadX(
    file_pth: Path, loadFunc, num_instances: int, num_features: int, useGpu=False,
) -> np.ndarray:
    """
    Load data for training/testing

    Parameters
    ----------
    file_pth
        Path to the file containing the data
    loadFunc
        The function to load the data (loadNpyFile or loadIdxFile)
    num_instances
        (Expected) Number of instances (rows) in the data
    num_features
        (Expected) Number of features (columns) in the data
    useGpu
        Whether to use GPU or CPU as data device

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_instances, num_features) and dtype np.float32
    """

    X = (
        loadFunc(file_pth, False, useGpu)
        .reshape(num_instances, num_features)
        .astype(np.float32)
    )

    # using inplace operator to not waste memory on copying and operating on a copy
    X /= 255.0
    if useGpu:
        cp.cuda.Stream.null.synchronize()

    return X


def loadY(file_pth: Path, loadFunc, useGpu=False, encoded=True) -> np.ndarray:
    """
    Load labels for training/testing

    Parameters
    ----------
    file_pth
        Path to the file containing the data
    loadFunc
        The function to load the data (loadNpyFile or loadIdxFile)
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the labels

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_instances, num_classes or None) and dtype np.int8

    """

    y = loadFunc(file_pth, True, useGpu).astype(np.int8)

    if encoded:
        y = oneHotEncode(MNIST_CLASSES, y)

    if useGpu:
        cp.cuda.Stream.null.synchronize()

    return y


def loadIdxTraining(dataDir: Path, useGpu=False, encoded=True) -> XY_DATA:
    """
    Loads data for training from dir with IDX3 files.

    Parameters
    ----------
    dataDir
        Path to the directory containing train-(images | labels)-idx3-ubyte
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the labels

    Returns
    -------
    XY_DATA
        A 2-element tuple of numpy arrays (X,y) |
        X: Shape = (instances, features), dtype = np.float32 |
        y: Shape = (instances, num_classes or 1), dtype = np.uint8
    """

    xPth: Path = dataDir / "train-images-idx3-ubyte"
    yPth: Path = dataDir / "train-labels-idx1-ubyte"

    assert xPth.exists()
    assert yPth.exists()

    instances = 60000
    features = 784

    X = loadX(xPth, loadIdxFile, instances, features, useGpu=useGpu)
    y = loadY(yPth, loadIdxFile, useGpu=useGpu, encoded=encoded)

    return X, y


def loadIdxTesting(dataDir: Path, useGpu=False) -> XY_DATA:
    """
    Loads data for testing from dir with IDX3 files.

    Parameters
    ----------
    dataDir
        Path to the directory containing t10k-(images | labels)-idx3-ubyte
    useGpu
        Whether to use GPU or CPU as data device

    Returns
    -------
    XY_DATA
        A 2-element tuple of numpy arrays (X,y) |
        X: Shape = (instances, features), dtype = np.float32 |
        y: Shape = (instances, 1), dtype = np.uint8

    """

    xPth: Path = dataDir / "t10k-images-idx3-ubyte"
    yPth: Path = dataDir / "t10k-labels-idx1-ubyte"

    assert xPth.exists()
    assert yPth.exists()

    instances = 10000
    features = 784

    X = loadX(xPth, loadIdxFile, instances, features, useGpu=useGpu)
    y = loadY(yPth, loadIdxFile, useGpu=useGpu, encoded=False).astype(np.uint8)

    if y.ndim == 2:
        assert y.shape[1] == 1
        y = y.squeeze()

    return X, y


def loadNpyTraining(
    dataDir: Path, instances: int, features: int, useGpu=False, encoded=True
) -> XY_DATA:
    """
    Loads data for training from dir with NPY files.

    Parameters
    ----------
    dataDir
        Path to the directory containing data
    instances
        Number of dataset instances (rows)
    features
        Number of features (columns)
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the labels

    Returns
    -------
    XY_DATA
        A 2-element tuple of numpy arrays (X,y) |
        X: Shape = (instances, features), dtype = np.float32 |
        y: Shape = (instances, num_classes or 1), dtype = np.uint8
    """

    xPth = dataDir / "train_images.npy"
    yPth = dataDir / "train_labels.npy"

    assert xPth.exists()
    assert yPth.exists()

    X = loadX(xPth, loadNpyFile, instances, features, useGpu=useGpu)
    y = loadY(yPth, loadNpyFile, useGpu=useGpu, encoded=encoded)

    return X, y


def loadNpyTesting(
    dataDir: Path, instances: int, features: int, useGpu=False
) -> XY_DATA:
    """
    Loads data for testing from dir with NPY files.

    Parameters
    ----------
    dataDir
        Path to the directory containing data
    instances
        Number of dataset instances (rows)
    features
        Number of features (columns)
    useGpu
        Whether to use GPU or CPU as data device

    Returns
    -------
    XY_DATA
        A 2-element tuple of numpy arrays (X,y) |
        X: Shape = (instances, features), dtype = np.float32 |
        y: Shape = (instances, 1), dtype = np.uint8
    """

    xPth = dataDir / "test_images.npy"
    yPth = dataDir / "test_labels.npy"

    assert xPth.exists()
    assert yPth.exists()

    X = loadX(xPth, loadNpyFile, instances, features, useGpu=useGpu)
    y = loadY(yPth, loadNpyFile, useGpu=useGpu, encoded=False).astype(np.uint8)

    if y.ndim == 2:
        assert y.shape[1] == 1
        y = y.squeeze()

    return X, y


# SECTION: DATASET LOADERS


def loadMnistC(category: DATASETS, useGpu=False, encoded=True) -> TRAIN_TEST_DATA:
    """
    Loads MNIST-C data

    Parameters
    ----------
    category
        Dataset category
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the training labels (trainY)

    Returns
    -------
    TRAIN_TEST_DATA
        A 4-element tuple of numpy arrays (trainX, trainY, testX, testY) |
        trainX: Shape = (train_instances, features), dtype = np.float32 |
        trainY: Shape = (train_instances, num_classes or 1), dtype = np.uint8 |
        testX: Shape = (test_instances, features), dtype = np.float32 |
        testY: Shape = (test_instances, 1), dtype = np.uint8
    """

    val = str(category)
    assert val.startswith("mnist_c")
    subCat = val.split("-")[-1]
    dirPth: Path = MNISTCDIR / subCat

    assert dirPth.exists()
    assert dirPth.is_dir()

    trainInstances = 60000
    testInstances = 10000
    features = 784

    trainX, trainY = loadNpyTraining(
        dirPth, trainInstances, features, useGpu=useGpu, encoded=encoded
    )
    testX, testY = loadNpyTesting(dirPth, testInstances, features, useGpu=useGpu)

    return trainX, trainY, testX, testY


def loadMnist(useGpu=False, encoded=True) -> TRAIN_TEST_DATA:
    """
    Loads MNIST data

    Parameters
    ----------
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the training labels (trainY)

    Returns
    -------
    TRAIN_TEST_DATA
        A 4-element tuple of numpy arrays (trainX, trainY, testX, testY) |
        trainX: Shape = (train_instances, features), dtype = np.float32 |
        trainY: Shape = (train_instances, num_classes or 1), dtype = np.uint8 |
        testX: Shape = (test_instances, features), dtype = np.float32 |
        testY: Shape = (test_instances, 1), dtype = np.uint8
    """

    trainX, trainY = loadIdxTraining(MNISTDIR, useGpu=useGpu, encoded=encoded)
    testX, testY = loadIdxTesting(MNISTDIR, useGpu=useGpu)

    return trainX, trainY, testX, testY


def loadFashionMnist(useGpu=False, encoded=True) -> TRAIN_TEST_DATA:
    """
    Loads Fashion-MNIST data

    Parameters
    ----------
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the training labels (trainY)

    Returns
    -------
    TRAIN_TEST_DATA
        A 4-element tuple of numpy arrays (trainX, trainY, testX, testY) |
        trainX: Shape = (train_instances, features), dtype = np.float32 |
        trainY: Shape = (train_instances, num_classes or 1), dtype = np.uint8 |
        testX: Shape = (test_instances, features), dtype = np.float32 |
        testY: Shape = (test_instances, 1), dtype = np.uint8
    """

    trainX, trainY = loadIdxTraining(FASHIONMNISTDIR, useGpu=useGpu, encoded=encoded)
    testX, testY = loadIdxTesting(FASHIONMNISTDIR, useGpu=useGpu)

    return trainX, trainY, testX, testY


def loadAffNist(useGpu=False, encoded=True) -> TRAIN_TEST_DATA:
    root = DATADIR / "affnist"
    assert root.exists()

    prefix = "affNIST"
    train_instances = 1600000
    test_validation_instances = 320000
    features = 1600

    trainX = loadX(
        root / f"{prefix}_trainX.npy",
        loadNpyFile,
        train_instances,
        features,
        useGpu=useGpu,
    )
    trainY = loadY(
        root / f"{prefix}_trainY.npy", loadNpyFile, useGpu=useGpu, encoded=encoded
    )
    testX = loadX(
        root / f"{prefix}_testX.npy",
        loadNpyFile,
        test_validation_instances,
        features,
        useGpu=useGpu,
    )
    testY = loadY(
        root / f"{prefix}_testY.npy", loadNpyFile, useGpu=useGpu, encoded=False
    ).astype(np.uint8)

    return trainX, trainY, testX, testY


def loadCifar10(useGpu=False, encoded=True) -> TRAIN_TEST_DATA:
    """
    Loads CIFAR-10 (greyscale) data

    Parameters
    ----------
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the training labels (trainY)

    Returns
    -------
    TRAIN_TEST_DATA
        A 4-element tuple of numpy arrays (trainX, trainY, testX, testY) |
        trainX: Shape = (train_instances, features), dtype = np.float32 |
        trainY: Shape = (train_instances, num_classes or 1), dtype = np.uint8 |
        testX: Shape = (test_instances, features), dtype = np.float32 |
        testY: Shape = (test_instances, 1), dtype = np.uint8
    """

    root = DATADIR / "cifar-10"
    assert root.exists()

    prefix = "cifar-10_greyscale"
    train_instances = 50000
    test_validation_instances = 10000
    features = 1024

    trainX = loadX(
        root / f"{prefix}_trainX.npy",
        loadNpyFile,
        train_instances,
        features,
        useGpu=useGpu,
    )
    trainY = loadY(
        root / f"{prefix}_trainY.npy", loadNpyFile, useGpu=useGpu, encoded=encoded
    )
    testX = loadX(
        root / f"{prefix}_testX.npy",
        loadNpyFile,
        test_validation_instances,
        features,
        useGpu=useGpu,
    )
    testY = loadY(
        root / f"{prefix}_testY.npy", loadNpyFile, useGpu=useGpu, encoded=False
    ).astype(np.uint8)

    return trainX, trainY, testX, testY


LOADING_FUNCS = {
    DATASETS.mnist: loadMnist,
    DATASETS.fashion: loadFashionMnist,
    DATASETS.cifar10: loadCifar10,
}


def loadDataset(
    dataset: DATASETS,
    useGpu=False,
    encoded=True,
    quant_precision: int = None,
    quantize_test=False,
) -> TRAIN_TEST_DATA:
    """
    Loads a given dataset

    Parameters
    ----------
    dataset
        One of the available datasets (in the enum DATASETS)
    useGpu
        Whether to use GPU or CPU as data device
    encoded
        Whether to one-hot encode the training labels (trainY)
    quant_precision
        The precision (in bits) for quantization

    Returns
    -------
    TRAIN_TEST_DATA
        A 4-element tuple of numpy arrays (trainX, trainY, testX, testY) |
        trainX: Shape = (train_instances, features), dtype = np.float32 |
        trainY: Shape = (train_instances, num_classes or 1), dtype = np.uint8 |
        testX: Shape = (test_instances, features), dtype = np.float32 |
        testY: Shape = (test_instances, 1), dtype = np.uint8
    """

    if str(dataset).startswith("mnist_c"):
        data = loadMnistC(dataset, useGpu=useGpu, encoded=encoded)
    else:
        loadFunc = LOADING_FUNCS[dataset]
        data = loadFunc(useGpu=useGpu, encoded=encoded)

    if quant_precision is not None:
        assert type(quant_precision) is int
        assert quant_precision > 1

        trainX, trainY, testX, testY = data
        quantize(trainX, precision=quant_precision)
        if quantize_test:
            quantize(testX, precision=quant_precision)
        data = (trainX, trainY, testX, testY)

    return data


if __name__ == "__main__":
    trainX, trainY, testX, testY = loadDataset(DATASETS.mnistc_jpeg_compression)
    print(testY.shape)
    # mnistCArr = [x for x in DATASETS if str(x).startswith("mnist_c")]
    # totalDatasets = len(mnistCArr)
    # for ds in mnistCArr:
    #     trainX, trainY, testX, testY = loadDataset(ds)
    #     print("=" * 5)
    #     print(ds)
    #     print((trainX.shape, trainX.dtype))
    #     print((trainY.shape, trainY.dtype))
    #     print((testX.shape, testX.dtype))
    #     print((testY.shape, testY.dtype))
    #
    # print("-" * 15)
    # print(f"Total DS: {totalDatasets}")
