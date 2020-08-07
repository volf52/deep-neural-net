from mlpcode.network import Network
from mlpcode.utils import DATASETS, MODELDIR, loadDataset, normalize
from mlpcode.activation import ActivationFuncs as af
from mlpcode.callbacks import ErrorCallback, ImageErrorCallback
import pickle
from time import time

useGpu = True
binarized = True
ds = DATASETS.mnist


_, _, testX, testY = loadDataset(ds, useGpu=useGpu)
del _

st = time()

lst = []
for i in range(8):
    nn = Network.fromModel(MODELDIR / f"{i}.hdf5", useGpu=useGpu, binarized=binarized)
    nn.compile(hiddenAf=af.sign, outAf=af.identity)
    p = 0.14
    icb = ImageErrorCallback(p)
    runningSum = 0.
    for _ in range(100):
        newX = icb(testX, gpu=useGpu)
        newX = normalize(newX, newMin=-1, newMax=1)
        acc = nn.get_accuracy(newX, testY)
        runningSum += acc
    totalAcc = runningSum / 100
    print(f"p: {p} \tAcc: {totalAcc:0.02f}\t")
    lst.append(totalAcc)

end = time()

print(f'\n\nTotalTime: {end-st:0.5f}')


# nn = Network.fromModel(modelPth, useGpu=useGpu, binarized=binarized)
# nn.compile(hiddenAf=af.sign, outAf=af.softmax)
# acc = nn.get_accuracy(testX, testY)
# print(f"P: None\tAcc: {acc:0.02f}")
#
# cb = ErrorCallback(0.1, mode=2, bnn=binarized)
# nn.addCallbacks(cb)
# acc = nn.get_accuracy(testX, testY)
# print(f"P: 0.05\tAcc: {acc:0.02f}")
#
# nn.clearCallbacks()
#
# cb = ErrorCallback(0.2, mode=2, bnn=binarized)
# nn.addCallbacks(cb)
# acc = nn.get_accuracy(testX, testY)
# print(f"P: 0.2\tAcc: {acc:0.02f}\n\n")