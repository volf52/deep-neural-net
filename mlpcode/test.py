from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import DATASETS, loadDataset, MODELDIR

useGpu = True
dataset = DATASETS.mnist
print("Loading {}".format(dataset))
trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu)
print("Finished loading {} data".format(dataset))
layers = [trainX.shape[1], 150, 300, 10]
epochs = 50
batchSize = 50
lr = 7e-2
print("Creating neural net")

# Creating from scratch
nn = Network(
    layers,
    useGpu=useGpu,
    hiddenAf=af.sigmoid,
    outAf=af.softmax,
    lossF=lf.cross_entropy,
    binarized=False,
)

# Save must be the name of the dataset, if we want to save the model
# nn.train(trainX, trainY, epochs, batch_size=batchSize, lr=lr, valX=testX, valY=testY, save=dataset)



# Creating from a pretrained model
modelPath = MODELDIR / "mnist_fp.npz"
assert modelPath.exists()
nn = Network.fromModel(
    modelPath,
    useGpu=useGpu,
    hiddenAf=af.sigmoid,
    outAf=af.softmax,
    lossF=lf.cross_entropy,
    binarized=True
)
nn.train(trainX, trainY, 30, batch_size=batchSize, lr=lr, valX=testX, valY=testY, save=dataset)