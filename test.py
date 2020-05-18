from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LossFuncs as lf
from mlpcode.network import Network
from mlpcode.utils import DATASETS, loadDataset

useGpu = False
binarized = False
dataset = DATASETS.fashion
print("Loading {}".format(dataset))
trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu)
# Set quant_precision to any integer > 1 to quantize the input,
# The quantized input having quant_precision + 1 unique elements
# trainX, trainY, testX, testY = loadDataset(dataset, useGpu=useGpu, quant_precision=2)
print("Finished loading {} data".format(dataset))

layers = [trainX.shape[1], 512, 10]
epochs = 100
batchSize = 600
lr = 0.07
# lr = LRScheduler(alpha=0.07, decay_rate=0.8, strategy=LRS.time)

print("\nCreating neural net")
# Creating from scratch
nn = Network(layers, useGpu=useGpu, binarized=binarized)

# Creating from a pretrained model
# modelPath = MODELDIR / "mnist_1589624361.944349.npz"
# assert modelPath.exists()
# nn = Network.fromModel(modelPath, useGpu=useGpu, binarized=binarized)

# Must compile the model before trying to train it
nn.compile(lr=lr, hiddenAf=af.leaky_relu, outAf=af.softmax, lossF=lf.cross_entropy)

# Save best will switch the model weights and biases to the ones with best accuracy at the end of the training loop
nn.train(trainX, trainY, epochs, batch_size=batchSize, save_best_params=True)

# Save will be called separately if we want to save the model
# Set binarized to true if you want to save the binary version of the weights
nn.save_weights(modelName=str(dataset), binarized=False)

# Change binarized to true to get the accuracy using binarized weights
testingIsBinarized = False
correct = nn.evaluate(testX, testY, batch_size=batchSize, binarized=testingIsBinarized)
acc = correct / testX.shape[0] * 100.0
print(
    "Accuracy on test set with {3} testing:\t{0} / {1} : {2:.03f}%".format(
        correct, testX.shape[0], acc, ["normal", "binarized"][testingIsBinarized]
    )
)
