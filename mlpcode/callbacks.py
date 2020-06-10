import numpy as np
import cupy as cp


class Callback:
    pass


class ErrorCallback(Callback):
    def __init__(self, n_bits: int, p: float, mode=0, bnn=False):
        assert mode in tuple(range(3))
        super(ErrorCallback, self).__init__()
        self.nbits = n_bits
        self.p = p
        self.mode = mode
        self.forBnn = bnn

    def _flipFunc(self, inp):
        assert inp.size == 32

        idx = np.arange(0, 32, dtype=np.int8)

        # If the MSB of the exponent field is high, ignore the rest of bits in the exponent field
        # The reason this bit comes at 25 instead of 1, is the little endian representation
        if inp[25]:
            idx[26:32] = -1
            idx[16] = -1
        # Otherwise, ignore the MSB itself
        else:
            idx[25] = -1

        if self.mode == 0:
            idx[inp[idx]] = -1
        elif self.mode == 1:
            idx[~inp[idx]] = -1

        idx = idx[idx != -1]
        idx = np.random.choice(idx, min(idx.size, self.nbits), replace=False)
        inp[idx] = ~inp[idx]

        return inp

    def _flipBNN(self, inp):
        if self.mode == 2:
            inp = -1 * inp

        else:
            if self.mode == 0:
                # -1/negative represent 0 in BNN case
                selector = inp < 0
            else:
                # +1/positive represent 1 in BNN (this is tentative)
                selector = inp > 0

            inp[selector] = -inp[selector]

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

        flipped = np.apply_along_axis(self._flipFunc, 1, unpacked)

        packed = self.repack(flipped)

        return packed

    def __call__(self, inp: np.ndarray, gpu=False):
        assert inp.ndim == 2

        if gpu:
            inp = cp.asnumpy(inp)

        shape = inp.shape
        inpFlat = np.array(inp.flatten())
        idxArr = np.random.choice(inpFlat.size, round(inpFlat.size * self.p), replace=False)

        if idxArr.size == 0:
            return inp

        print(f"Flipping bits for {len(idxArr)} / {inpFlat.size} values")
        flipFunc = [self._flip, self._flipBNN][self.forBnn]
        inpFlat[idxArr] = flipFunc(inpFlat[idxArr])

        inp = inpFlat.reshape(shape)

        if gpu:
            inp = cp.array(inp)

        return inp


if __name__ == "__main__":
    w = np.random.randn(3, 5).astype(np.float32)

    err = ErrorCallback(2, 0.2, mode=0)

    print(w)
    flippedW = err(w)
    print(flippedW)