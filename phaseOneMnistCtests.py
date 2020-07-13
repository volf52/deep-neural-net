from mlpcode.network import Network
from mlpcode.activation import ActivationFuncs as af
from mlpcode.utils import DATASETS, loadDataset, MODELDIR
from mlpcode.callbacks import ErrorCallback


useGpu = False
binarized = True
ds = DATASETS.fashion

modelPath = MODELDIR / "bnn_fashion_mnist_1024_1024_512_yesBN_noBias_89_750_995_89_56.hdf5"
assert modelPath.exists()

# nn= Network([784, 256, 10], useGpu=useGpu, useBatchNorm=True)
nn = Network.fromModel(modelPath, useGpu=useGpu, binarized=binarized)

# errorCb = ErrorCallback(3, 0.2, mode=2, bnn=binarized)
# nn.addCallbacks(errorCb, num_layers=1)

nn.compile(hiddenAf=af.sign, outAf=af.identity)

_, _, testX, testY = loadDataset(ds, useGpu=useGpu)

acc = nn.get_accuracy(testX, testY)
print(f"{ds}:\t{acc:0.2f} %")


