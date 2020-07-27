import pickle

from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.optim import LRScheduler, LRSchedulerStrat as LRS
from mlpcode.utils import DATASETS, loadDataset, MODELDIR, split_train_valid, normalize

useGpu = True
binarized = True
dataset = DATASETS.mnist
print("Loading {}".format(dataset))
trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu)

trainX = normalize(trainX)
testX = normalize(testX)

trainX, valX, trainY, valY = split_train_valid(trainX, trainY)

print("Finished loading {} data".format(dataset))

layers = [trainX.shape[1], 500, 500, 10]
epochs = 1000
batchSize = 200
lrStart = 0.03
lrEnd = 3e-7
lr = 0.01
# lr = LRScheduler(
#     alpha=lrStart, decay_rate=(lrStart - lrEnd) ** (1 / epochs), strategy=LRS.exp
# )

print("\nCreating neural net")
# Creating from scratch
nn = Network(
    layers, useGpu=useGpu, binarized=binarized, useBatchNorm=True, useBias=False
)

# Creating from a pretrained model
# modelPath = MODELDIR / "bnnKeras.hdf5"
# assert modelPath.exists()
# nn = Network.fromModel(modelPath, useGpu=useGpu, binarized=binarized)

# Must compile the model before trying to train it
nn.compile(lr=lr, hiddenAf=af.sign, outAf=af.softmax, lossF=lf.cross_entropy)

# Save best will switch the model weights and biases to the ones with best accuracy at the end of the training loop
history = nn.train(
    trainX,
    trainY,
    epochs,
    batch_size=batchSize,
    save_best_params=True,
    valX=valX,
    valY=valY,
)

# unitsListStr = "_".join(map(str, nn.unitList))
# modelName = f"bnn_{dataset}_{unitsListStr}_bnnNew"
# nn.save_weights(modelName=modelName, binarized=False)
# with open(MODELDIR / f"{modelName}_history.pkl", "wb") as fp:
#     pickle.dump(history, fp)

correct = nn.evaluate(testX, testY)
acc = correct / testX.shape[0] * 100.0
print(
    "Accuracy on test set:\t{0} / {1} : {2:.03f}%".format(correct, testX.shape[0], acc)
)
