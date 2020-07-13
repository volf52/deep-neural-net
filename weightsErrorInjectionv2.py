from mlpcode.network import Network
from mlpcode.utils import DATASETS, MODELDIR, loadDataset
from mlpcode.activation import ActivationFuncs as af
from mlpcode.callbacks import ErrorCallback

useGpu = False
binarized = True
ds = DATASETS.mnist

modelPth = MODELDIR / "bnn_mnist_1024_1024_yesBN_noBias_98_433_968_98_27.hdf5"
assert modelPth.exists()

_, _, testX, testY = loadDataset(ds, useGpu=useGpu)
# pList = [0.01, 0.02, 0.05, 0.1, 0.2]
pList = [0.2]

for p in pList:
    lst = []
    # for _ in range(20):
    nn = Network.fromModel(modelPth, useGpu=useGpu, binarized=binarized)
    cb = ErrorCallback(p, mode=0, bnn=binarized)
    nn.compile(hiddenAf=af.sign, outAf=af.identity)
    nn.addCallbacks(cb)
    acc = nn.get_accuracy(testX, testY)
    lst.append(acc)

    acc = sum(lst) / len(lst)
    print(f"P: {p:0.02f}\tAcc: {acc:0.02f}\n\n")
