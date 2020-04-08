from typing import List

import cupy as cp
import numpy as np

from mlpcode.activation import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS
from mlpcode.activation import ActivationFuncs as af


class NeuralNet:
    def __init__(
        self,
        numNeurons: List[int],
        outClasses: int,
        hidden_activation: af = af.relu,
        output_activation=None,
    ):

        assert len(numNeurons) > 1
        assert outClasses > 1

        self._memory = {}
        self.numNeurons = numNeurons
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

    def init_weights(self):
        # Will change to xavier init later
        nLen = len(self.arch)
        params = {}

        for i, layer in enumerate(self.arch):
            layerIdx = i + 1
            layerInp = layer["input_dim"]
            layerOut = layer["out_dim"]

            params["W" + str(layerIdx)] = (
                cp.random.randn(layerOut, layerInp) * 0.1
            )
            params["b" + str(layerIdx)] = cp.random.randn(layerOut, 1) * 0.1

        return params


if __name__ == "__main__":
    from pprint import pprint

    nLayers = [2, 4, 6, 6, 4]
    out = 2
    nn = NeuralNet(nLayers, out)
    pprint(nn.params)
