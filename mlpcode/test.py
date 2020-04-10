from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LOSS_DERIVATES, LOSS_FUNCS
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import read_test, read_train

useGpu = False
print("Loading data")
X_train, y_train = read_train(useGpu=useGpu)
X_test, y_test = read_test(useGpu=useGpu)
print("Finished loading mnist data")
layers = [784, 64, 10]
epochs = 500
print("Creating neural net")
nn = Network(
    layers,
    useGpu=useGpu,
    hiddenAf=af.relu,
    outAf=af.softmax,
    lossF=lf.cross_entropy,
)
nn.train(X_train, y_train, epochs, 600, 3.0, X_test, y_test)
