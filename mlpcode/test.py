import time
from math import sqrt

import cupy as cp
import numpy as np
from numba import vectorize

npoints = int(1e7)

x_cpu = np.arange(npoints, dtype=np.float32)
x_gpu = cp.arange(npoints, dtype=cp.float32)
cp.cuda.Stream.null.synchronize()

@vectorize
def cpu_sqrt(x):
    return sqrt(x)

@vectorize(['float32(float32)'], target='cuda')
def gpu_sqrt(x):
    return sqrt(x)

s = time.time()
cpu_sqrt(x_cpu)
e = time.time()
print(f"Time for cpu:\t\t{e - s}")


s = time.time()
np.sqrt(x_cpu)
e = time.time()
print(f"Time for numpy:\t\t{e - s}")

s = time.time()
gpu_sqrt(x_cpu)
e = time.time()
print(f"Time for numpy + numba:\t{e - s}")

s = time.time()
gpu_sqrt(x_gpu)
e = time.time()
print(f"Time for cupy + numba:\t{e - s}")
