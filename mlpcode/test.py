from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import DATASETS, loadDataset

useGpu = True
dataset = DATASETS.mnist
print("Loading {}".format(dataset))
trainX, trainY, testX, testY = loadDataset(dataset)
print("Finished loading {} data".format(dataset))
layers = [784, 500, 1000, 10]
epochs = 500
print("Creating neural net")
nn = Network(
    layers,
    useGpu=useGpu,
    hiddenAf=af.sign,
    outAf=af.softmax,
    lossF=lf.cross_entropy,
    binarized=True
)

nn.train(trainX, trainY, epochs, 100, 0.007, testX, testY)
