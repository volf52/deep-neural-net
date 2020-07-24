from mlpcode.network import Network
from mlpcode.utils import DATASETS, MODELDIR, loadDataset, normalize
from mlpcode.activation import ActivationFuncs as af
from mlpcode.callbacks import ErrorCallback

useGpu = False
binarized = True
ds = DATASETS.mnist

modelPth = MODELDIR / "bnnPytorchCheckpTest.hdf5"
assert modelPth.exists()

_, _, testX, testY = loadDataset(ds, useGpu=useGpu)
del _
testX = normalize(testX, newMin=-1., newMax=1.)
# pList = [0.01, 0.02, 0.05, 0.1, 0.2]
p = 0.1

nn = Network.fromModel(modelPth, useGpu=useGpu, binarized=binarized)
nn.compile(hiddenAf=af.sign, outAf=af.softmax)
acc = nn.get_accuracy(testX, testY)
print(f"P: None\tAcc: {acc:0.02f}")

cb = ErrorCallback(p, mode=2, bnn=binarized)
nn.addCallbacks(cb)
acc = nn.get_accuracy(testX, testY)
print(f"P: 0.1\tAcc: {acc:0.02f}")

nn.clearCallbacks()

cb = ErrorCallback(0.2, mode=2, bnn=binarized)
nn.addCallbacks(cb)
acc = nn.get_accuracy(testX, testY)
print(f"P: 0.2\tAcc: {acc:0.02f}\n\n")