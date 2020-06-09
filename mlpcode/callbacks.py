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

    @staticmethod
    def unpack(inp):
        buff = np.frombuffer(inp, dtype=np.uint8)
        unpacked = np.unpackbits(buff).astype(np.bool).reshape(-1, 32)
        return unpacked

    @staticmethod
    def repack(inp):
        buff = np.packbits(inp)
        packed = np.frombuffer(buff, dtype=np.float32)
        return packed


    def _flip(self, inp: np.ndarray):
        unpacked = self.unpack(inp)

        flipped = self._flipFunc(unpacked)

        packed = self.repack(flipped)

        return packed

    def __call__(self, inp: np.ndarray):
        assert inp.ndim == 2

        rowIdx = np.random.choice(inp.shape[0], round(inp.shape[0] * self.pFlip), replace=False)
        colIdx = np.random.choice(inp.shape[1], round(inp.shape[1] * self.pNeurons), replace=False)


        if rowIdx.size == 0 or colIdx.size == 0:
            return inp

        flipSeq = (rowIdx, colIdx)
        print(f"Flipping {flipSeq}")

        inp[flipSeq] = self._flip(inp[flipSeq])

        return inp


if __name__ == "__main__":
    w = np.random.randn(3, 5).astype(np.float32)

    err = ErrorCallback(2, 0.2, 0.3)

    print(w)
    flippedW = err(w)
    print(flippedW)