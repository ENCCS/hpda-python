import numpy as np
import numba
N = 50
A = np.random.rand(N,N)
B = np.random.rand(N,N)
C = np.random.rand(N,N)

d_A = numba.cuda.to_device(A)
d_B = numba.cuda.to_device(B)
d_C = numba.cuda.to_device(C)

TPB = 16
threadsperblock = (TPB, TPB)
blockspergrid = (16,16)

%timeit matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C); numba.cuda.synchronize()
# 90.9 µs ± 244 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
