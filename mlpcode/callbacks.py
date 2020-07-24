import numpy as np
import cupy as cp

class Callback:
    def __init__(self, p:float, mode: int):
        assert mode in (0, 1, 2)
        # mode 0: 0 -> 1
        # mode 1: 1 -> 0
        # mode 2: hybrid
        
        self.p = p
        self.mode = mode

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


class ErrorCallback(Callback):
    def __init__(self, p: float, mode=2, bnn=False):
        super(ErrorCallback, self).__init__(p, mode)
        self.forBnn = bnn
        # self.nBits = 1

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
        nBits = round(idx.size * self.p)
        idx = np.random.choice(idx, min(idx.size, nBits), replace=False)
        inp[idx] = ~inp[idx]

        return inp

    def _flipBNN(self, inp):
        flipSize = inp.size
        if self.mode == 2:
            inp = -1 * inp

        else:
            if self.mode == 0:
                # -1/negative represent 0 in BNN case
                selector = inp < 0
            else:
                # +1/positive represent 1 in BNN (this is tentative)
                selector = inp > 0
            flipSize = selector.sum()
            inp[selector] = -inp[selector]

        # print(f"{flipSize} values flipped")
        return inp

    def _flipBNNMask(self, inp: np.ndarray):
        out = inp.copy()
        mask = np.random.uniform(size=inp.shape) < self.p
        out[mask] = -out[mask]
        return out

    def _flip(self, inp: np.ndarray):
        unpacked = self.unpack(inp)

        flipped = np.apply_along_axis(self._flipFunc, 1, unpacked)

        packed = self.repack(flipped)

        return packed

    def __call__(self, inp: np.ndarray, gpu=False):
        assert inp.ndim == 2

        if self.forBnn:
            return self._flipBNNMask(inp)

        if gpu:
            inp = cp.asnumpy(inp)
        elif cp.get_array_module(inp) == cp:
            print("You forgot to set `gpu=True` for a Cupy array")
            gpu = True
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

class ImageErrorCallback(Callback):
    def __init__(self, p: float, mode=2):
        super(ImageErrorCallback, self).__init__(p, mode)
        self.nbits = self.p * 8

    def intError(self, arr: np.ndarray):
        assert arr.dtype == np.uint8

        unpacked = np.unpackbits(arr)

        nbits = round(self.p * unpacked.size)
        if self.mode == 0:
            chosen, = np.where(unpacked == 0)
            idx = np.random.choice(chosen, min(chosen.size, nbits), replace=False)
        elif self.mode == 1:
            chosen, = np.where(unpacked == 1)
            idx = np.random.choice(chosen, min(chosen.size, nbits), replace=False)
        else:
            idx = np.random.choice(unpacked.size, nbits, replace=False)

        unpacked[idx] = np.logical_not(unpacked[idx])

        return np.packbits(unpacked)

    def __call__(self, arr: np.ndarray, gpu=False):
        if gpu:
            arr = cp.asnumpy(arr)
        elif cp.get_array_module(arr) == cp:
            print("You forgot to set `gpu=True` for a Cupy array")
            gpu = True
            arr = cp.asnumpy(arr)

        if arr.dtype == np.uint8:
            arr = np.apply_along_axis(self.intError, 1, arr)
        # elif arr.dtype == np.float32:
        #     result = self.floatError(arr)
        else:
            raise ValueError("Only uint8 allowed")

        if gpu:
            arr = cp.array(arr)

        return arr

if __name__ == "__main__":
    from mlpcode.utils import loadDataset, DATASETS
    _, _, testX, _ = loadDataset(DATASETS.mnist)

    icb = ImageErrorCallback(0.1)

    print(testX[:10,:])
    testX = icb(testX)
    print(testX[:10, :])