from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.optim import LRScheduler, LRSchedulerStrat as LRS
from mlpcode.utils import DATASETS, loadDataset, MODELDIR, split_train_valid

useGpu = True
binarized = False
dataset = DATASETS.mnist
print("Loading {}".format(dataset))
trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu)
trainX, valX, trainY, valY = split_train_valid(trainX, trainY)
# Set quant_precision to any integer > 1 to quantize the input,
# The quantized input having quant_precision + 1 unique elements
# trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu, quant_precision=2)
print("Finished loading {} data".format(dataset))

layers = [trainX.shape[1], 512, 512, 10]
epochs = 1000
batchSize = 100
lrStart = 7e-2
lrEnd = 1e-4
# lr = 0.07
lr = LRScheduler(
    alpha=lrStart, decay_rate=(lrStart - lrEnd) ** (1 / epochs), strategy=LRS.exp
)

print("\nCreating neural net")
# Creating from scratch
nn = Network(
    layers, useGpu=useGpu, binarized=binarized, useBatchNorm=False, useBias=True
)

# Creating from a pretrained model
# modelPath = MODELDIR / "bnnKeras.hdf5"
# assert modelPath.exists()
# nn = Network.fromModel(modelPath, useGpu=useGpu, binarized=binarized)

# Must compile the model before trying to train it
nn.compile(lr=lr, hiddenAf=af.sigmoid, outAf=af.softmax, lossF=lf.cross_entropy)

# Save best will switch the model weights and biases to the ones with best accuracy at the end of the training loop
nn.train(
    trainX,
    trainY,
    epochs,
    batch_size=batchSize,
    save_best_params=True,
    valX=valX,
    valY=valY,
)

# Save will be called separately if we want to save the model
# Set binarized to true if you want to save the binary version of the weights
unitsListStr = "_".join(map(str, nn.unitList))
nn.save_weights(modelName=f"{dataset}_{unitsListStr}", binarized=False)

correct = nn.evaluate(testX, testY, batch_size=-1)
acc = correct / testX.shape[0] * 100.0
print(
    "Accuracy on test set:\t{0} / {1} : {2:.03f}%".format(correct, testX.shape[0], acc)
)
