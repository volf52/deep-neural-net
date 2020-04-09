import struct
from pathlib import Path

import cupy as cp
import numpy as np

DATA_DIR: Path = Path(__file__).parent / "data"


def read_idx(filename, xp, custom_shape=None):
    with open(filename, "rb") as f:
        zero, data_type, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for d in range(dims))
        if shape is not None:
            shape = custom_shape

        # Added division by 255.0 to normalize the data
        return xp.fromfile(f, dtype=np.uint8).reshape(shape) / 255.0


def read_tenk(useGpu=True, reshape=False):
    tenkPath = DATA_DIR / "t10k-images.idx3-ubyte"
    tenkLabels = DATA_DIR / "t10k-labels.idx1-ubyte"
    xShape = None
    yShape = None
    if not (tenkPath.exists() and tenkLabels.exists()):
        raise FileNotFoundError(
            f"10k-images.idx3-ubyte or its labels not found"
        )

    if useGpu:
        xp = cp
    else:
        xp = np

    if reshape:
        xShape = (-1, 784)
        yShape = (-1, 1)

    data = read_idx(tenkPath, xp, xShape)
    labels = read_idx(tenkLabels, xp, yShape)

    return data, labels


def read_train(useGpu=True, reshape=False):
    trainPath = DATA_DIR / "train-images.idx3-ubyte"
    trainlabels = DATA_DIR / "train-labels.idx1-ubyte"
    xShape = None
    yShape = None
    if not (trainPath.exists() and trainlabels.exists()):
        raise FileNotFoundError(
            f"train-images.idx3-ubyte or its labels not found"
        )

    if useGpu:
        xp = cp
    else:
        xp = np

    if reshape:
        xShape = (-1, 784)
        yShape = (-1, 1)

    data = read_idx(trainPath, xp, xShape)
    labels = read_idx(trainlabels, xp, yShape)

    return data, labels


if __name__ == "__main__":
    tenk, tlabels = read_tenk(reshape=True)
    print(tenk.shape)
    print(tlabels.shape)
    train, trLabels = read_train(reshape=True)
    print(train.shape)
    print(trLabels.shape)
