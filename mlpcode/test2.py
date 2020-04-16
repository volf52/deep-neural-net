import csv
from pathlib import Path

import cupy as cp

from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import oneHotEncoding


def readData(dataFile: Path):
    X = list()
    y = list()
    with open(dataFile, "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        for row in reader:
            x = list(map(float, row[:4]))
            X.append(x)
            y.append(int(row[4]))
    return cp.array(X), oneHotEncoding(2, cp.array(y))


X, y = readData(
    Path(__file__).parent / "data/data_banknote_authentication.txt"
)
nn = Network([4, 4, 2], useGpu=True, hiddenAf=af.leaky_relu)

nn.train(X, y, 50)
