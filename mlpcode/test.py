from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import loadMnist, loadFashionMnist

useGpu = True
print("Loading data")
trainX, trainY, testX, testY = loadMnist()
print("Finished loading mnist data")
layers = [784, 64, 10]
epochs = 500
print("Creating neural net")
# Don't use cross entropy until I include a method to turn Y labels to one-hot-encoded vectors
nn = Network(
    layers,
    useGpu=useGpu,
    hiddenAf=af.leaky_relu,
    outAf=af.softmax,
    lossF=lf.cross_entropy,
)

nn.train(trainX, trainY, epochs, 600, 1e-3, testX, testY)
