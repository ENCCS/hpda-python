import numpy as np
import numba
N = 50
A = np.random.rand(N,N)
B = np.random.rand(N,N)
C = np.random.rand(N,N)

TPB = 16
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(C.shape[1] / threadsperblock[1]))
#blockspergrid = (16,16)

%timeit matmul_kernel[blockspergrid, threadsperblock](A, B, C); numba.cuda.synchronize()
# 914 µs ± 869 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
