from typing import List, Dict, Union

import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af
from mlpcode.loss import LOSS_FUNCS, LOSS_DERIVATES
from mlpcode.loss import LossFuncs as lf

NDARR = Union[np.ndarray, cp.ndarray]


class NeuralNet:
    def __init__(
        self,
        numNeurons: List[int],
        outClasses: int,
        useGpu=True,
        hidden_activation: af = af.relu,
        output_activation=None,
    ):

        assert len(numNeurons) > 1
        assert outClasses > 1

        self._memory: Dict[str, NDARR] = {}
        self.numNeurons = numNeurons

        if useGpu:
            self.xp = cp
        else:
            self.xp = np

        if output_activation is None:
            if outClasses == 2:
                self.outAf = af.sigmoid
            else:
                self.outAf = af.softmax
        else:
            self.outAf = output_activation

        if outClasses == 2:
            outNeurons = 1
        else:
            outNeurons = outClasses

        self.arch = self.createArchitecture(
            numNeurons, outNeurons, hidden_activation, self.outAf
        )
        self.params = self.init_weights()

    @staticmethod
    def createArchitecture(
        numNeurons: List[int], outClasses: int, hiddenAf: af, outAf: af
    ):
        arch = [
            {
                "input_dim": numNeurons[i],
                "out_dim": numNeurons[i + 1],
                "activation": hiddenAf,
            }
            for i in range(len(numNeurons) - 1)
        ]
        arch.append(
            {
                "input_dim": numNeurons[-1],
                "out_dim": outClasses,
                "activation": outAf,
            }
        )
        return arch

    def init_weights(self) -> Dict[str, NDARR]:
        # Will change to xavier init later
        nLen = len(self.arch)
        params = {}

        for i, layer in enumerate(self.arch):
            layerIdx = i + 1
            layerInp = layer["input_dim"]
            layerOut = layer["out_dim"]

            params["W" + str(layerIdx)] = (
                self.xp.random.randn(layerOut, layerInp) * 0.1
            )
            params["b" + str(layerIdx)] = (
                self.xp.random.randn(layerOut, 1) * 0.1
            )

        return params

    def singleLayerForward(
        self, A_prev, W_curr, b_curr, activation: af = af.relu
    ):
        Z_curr = self.xp.dot(W_curr, A_prev) + b_curr

        afunc = ACTIVATION_FUNCTIONS.get(activation)
        if afunc is None:
            raise ValueError(f"Non-supported function: {activation}")

        return afunc(Z_curr), Z_curr

    def singleLayerBack(
        self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation: af = af.relu
    ):
        xp = cp.get_array_module(A_prev)
        m = A_prev.shape[1]
        afunc_deriv = ACTIVATION_DERIVATIVES.get(activation)
        if afunc_deriv is None:
            raise ValueError(f"Non-supported function: {activation}")

        dZ_curr = afunc_deriv(dA_curr, Z_curr)
        dW_curr = xp.dot(dZ_curr, A_prev.T) / m
        db_curr = xp.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = xp.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def forwardProp(self, Xi):
        self._memory = {}
        A_curr = Xi

        for i, layer in enumerate(self.arch):
            layerIdx = i + 1
            A_prev = A_curr

            afunc = layer["activation"]
            W_curr = self.params["W" + str(layerIdx)]
            b_curr = self.params["b" + str(layerIdx)]
            A_curr, Z_curr = self.singleLayerForward(
                A_prev, W_curr, b_curr, afunc
            )

            self._memory["A" + str(i)] = A_prev
            self._memory["Z" + str(layerIdx)] = Z_curr

        return A_curr

    def backProp(self, Y_hat, Y, lossFunc: lf) -> Dict[str, NDARR]:
        gradVals = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        loss_deriv = LOSS_DERIVATES.get(lossFunc)
        if loss_deriv is None:
            raise ValueError(f"Loss function not found: {lossFunc}")

        dA_prev = loss_deriv(Y_hat, Y)

        for prevLayerIdx, layer in reversed(list(enumerate(self.arch))):
            layerIdx = prevLayerIdx + 1
            curr_activation = layer["activation"]

            dA_curr = dA_prev

            A_prev = self._memory["A" + str(prevLayerIdx)]
            Z_curr = self._memory["Z" + str(layerIdx)]
            W_curr = self.params["W" + str(layerIdx)]
            b_curr = self.params["b" + str(layerIdx)]

            dA_prev, dW_curr, db_curr = self.singleLayerBack(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, curr_activation
            )

            gradVals["dW" + str(layerIdx)] = dW_curr
            gradVals["db" + str(layerIdx)] = db_curr

        return gradVals

    def updateWeights(self, gradVals, lr=0.01):
        for i, layer in enumerate(self.arch):
            layerIdx = i + 1
            self.params["W" + str(layerIdx)] -= (
                lr * gradVals["dW" + str(layerIdx)]
            )
            self.params["b" + str(layerIdx)] -= (
                lr * gradVals["db" + str(layerIdx)]
            )

    def train(self, X, Y, epochs, lossF: lf = lf.mse, lr=0.01, cont=False):
        if not cont:
            self.params = self.init_weights()

        lossFunc = LOSS_FUNCS.get(lossF)
        if lossFunc is None:
            raise ValueError(f"Loss function not found: {lossFunc}")

        cost_hist = []
        accuracy_hist = []

        for i in range(epochs):
            print("-" * 10)
            print(f"Epoch {i + 1}")
            Y_hat = self.forwardProp(X)
            cost = lossFunc(Y_hat, Y)
            cost_hist.append(cost)
            # accuracy = calcAccuracy(Y_hat, Y)
            # accuracy_hist.append(accuracy)

            gradVals = self.backProp(Y_hat, Y, lossF)
            self.updateWeights(gradVals, lr)

        return cost_hist, accuracy_hist


if __name__ == "__main__":
    from pprint import pprint
    from time import time

    nLayers = [3, 4, 6, 6, 3]
    out = 2

    # Transpose at the end to make the initial dot product possible.
    # Turn iterative dot products into a batch matrix operation
    X = cp.array([[2.4, 5.7, 3.6], [7.8, 8.9, 7.9]]).T
    Y = cp.array([[1.0], [0.0]]).T
    nn = NeuralNet(nLayers, out, output_activation=af.sigmoid, useGpu=True)
    pprint(nn.params)
    start = time()
    cl, al = nn.train(X, Y, 25, lossF=lf.mse)
    end = time()
    total_time_taken = end - start
    pprint(nn.params)
    print(f"Time per epoch is {total_time_taken / 5} seconds")
