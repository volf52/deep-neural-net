import numpy as np


class Callback:
    pass


class ErrorCallback(Callback):
    def __init__(self, n_bits: int, pFlip: float, pNeurons: float, mode=0):
        super(ErrorCallback, self).__init__()
        self.nbits = n_bits
        self.pFlip = pFlip
        self.pNeurons = pNeurons
        self.mode = mode

        if mode == 0:
            self._flipFunc = self._zeroToOne
        elif mode == 1:
            self._flipFunc = self._oneToZero
        elif mode == 3:
            self._flipFunc = self._hybridMode
        else:
            raise ValueError("Mode must have a value in range [0,2]")

    def _zeroToOne(self, inp):
        return inp

    def _oneToZero(self, inp):
        return inp

    def _hybridMode(self, inp):
        return inp

    def _flip(self, inp: np.ndarray):
        unpacked = np.unpackbits(inp)

        flipped = self._flipFunc(unpacked)

        packed = np.packbits(flipped)

        return packed

    def __call__(self, inp: np.ndarray):
        assert inp.ndim == 2

        rowIdx = np.random.choice(inp.shape[0], int(inp.shape[0] * self.pFlip), replace=False)
        colIdx = np.random.choice(inp.shape[1], int(inp.shape[1] * self.pNeurons), replace=False)

        flipSeq = (rowIdx, colIdx)

        inp[flipSeq] = self._flip(inp[flipSeq])

        return inp
