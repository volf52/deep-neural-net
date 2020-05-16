from mlpcode.network import Network
from SC.network import SCNetwork
from mlpcode.loss import LossFuncs as lf
from mlpcode.activation import ActivationFuncs as af
from mlpcode.utils import DATASETS, MODELDIR

if __name__ == "__main__":
    modelPath = MODELDIR / 'mnist_bnn_32.npz'

    num_instances = 1000
    precision = 256

    lr = 0.07
    hiddenAf = af.sigmoid
    outAf = af.softmax
    lossFunc = lf.cross_entropy

    useGpu = False
    binarized = True

    network = Network.fromModel(
        modelPath,
        useGpu=useGpu,
        binarized=binarized
    )

    SC_NN = SCNetwork(network, hiddenAf=hiddenAf, outAf=outAf, precision=precision, binarized=binarized)

    print(SC_NN.testDataset(DATASETS.mnist, num_instances=num_instances, parallel=True))






