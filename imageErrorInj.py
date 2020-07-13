from mlpcode.network import Network
from mlpcode.utils import DATASETS, MODELDIR, loadDataset, normalize
from mlpcode.activation import ActivationFuncs as af
from mlpcode.callbacks import ImageErrorCallback

useGpu = False
binarized = True
ds = DATASETS.mnist

modelPth = MODELDIR / "bnn_mnist_1024_1024_yesBN_noBias_98_433_968_98_27.hdf5"
assert modelPth.exists()

_, _, testX, testY = loadDataset(ds, useGpu=useGpu)

p = 0.05
icb = ImageErrorCallback(p)
testX = icb(testX)

testX = normalize(testX)

nn = Network.fromModel(modelPth, useGpu=useGpu, binarized=binarized)
nn.compile(hiddenAf=af.sign, outAf=af.identity)
acc = nn.get_accuracy(testX, testY)

print(f"P: {p:0.02f}\tAcc: {acc:0.02f}\n\n")
