import numpy as np
from numba import vectorize

from SC.utils import SCNumber, random_stream, bpe_encode


@vectorize(['boolean(boolean, boolean)'])
def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # return np.logical_not(np.logical_xor(x, y))
    return ~(x ^ y)


def scaled_add(x, y, scale=0.5):
    # x , y : 1D numpy array holding the bit-stream
    assert len(x) == len(y)
    n = len(x)
    selection = SCNumber(scale, precision=n)
    result = np.ndarray(n, dtype=np.bool)
    for i in range(n):
        if selection[i]:
            result[i] = x[i]
        else:
            result[i] = y[i]
    return result


def counter_add(x, y):
    # x , y : 1D numpy array holding the bit-stream
    # returns 1D numpy array representing the bit-stream of x + y
    assert len(x) == len(y)
    n = len(x)
    count = np.count_nonzero(x)
    count += np.count_nonzero(y)

    return SCNumber(2 * (count - n) / n, precision=n)


def sum(x: np.ndarray, conc=1):
    # x is 2D numpy array holding n bit-streams
    # returns 1D numpy array as one bit-stream
    assert len(x.shape) == 2
    n = len(x[0])
    count = np.count_nonzero(x)
    # decimal_value = (count - (len(x)*n - count) ) / n
    decimal_value = (2 * count - x.size) / n
    result = np.ndarray(conc * n, dtype=np.bool)
    i = 0
    while decimal_value > 1 and conc != 1:
        result[i:i + n] = 1
        decimal_value -= 1
        conc -= 1
        i += n

    while decimal_value < -1 and conc != 1:
        result[i:i + n] = 0
        decimal_value += 1
        conc -= 1
        i += n

    if -1 <= decimal_value <= 1:
        # The decimal value is already quantized,  No need to quantize it again
        result[i:i + n] = random_stream(n, int(bpe_encode(decimal_value) * n))
        if conc != 1:
            i += n
            result[i:i + n] = random_stream(n, n // 2)

    return result


def dot(x: np.ndarray, y: np.ndarray, conc=1):
    # x , y are 3D numpy arrays holding n*m bit-streams
    # returns 3D numpy array containing bit-streams representing the result of x dot y
    assert len(x.shape) == 3 and len(y.shape) == 3
    assert x.shape[2] == y.shape[2]  # checking precisions
    assert x.shape[1] == y.shape[0]  # (i,j) . (k,z) --> k=j

    result = np.ndarray((x.shape[0], y.shape[1], x.shape[2] * conc), dtype=np.bool)
    y_t = y.transpose(1, 0, 2)
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            result[i][j] = sum(mul(x[i], y_t[j]), conc=conc)

    return result