from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import DATASETS, loadDataset, MODELDIR

useGpu = True
dataset = DATASETS.mnistc_dotted_line
print("Loading {}".format(dataset))
trainX, trainY, testX, testY = loadDataset(dataset)
print("Finished loading {} data".format(dataset))
layers = [trainX.shape[1], 500, 10]
epochs = 100
batchSize = 600
lr = 0.01
print("Creating neural net")

# Creating from scratch
nn = Network(
    layers,
    useGpu=useGpu,
    hiddenAf=af.leaky_relu,
    outAf=af.softmax,
    lossF=lf.cross_entropy,
    binarized=False,
)

# Creating from a pretrained model
# modelPath = MODELDIR / "mnist_c-rotate_1588597666.073832.npz"
# assert modelPath.exists()
# nn = Network.fromModel(
#     modelPath,
#     useGpu=True,
#     hiddenAf=af.leaky_relu,
#     outAf=af.softmax,
#     lossF=lf.cross_entropy,
# )

# Save must be the name of the dataset, if we want to save the model
nn.train(trainX, trainY, epochs, batch_size=batchSize, lr=lr)
