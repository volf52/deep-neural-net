from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import read_test, read_train

useGpu = True
print("Loading data")
trainX, trainY = read_train(useGpu=useGpu)
# No need to one hot encod testY as we don't need it in that form
testX, testY = read_test(useGpu=useGpu, encoded=False)
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
