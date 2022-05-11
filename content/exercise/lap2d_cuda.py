import numpy as np
import numba
import numba.cuda

@numba.cuda.jit()
def lap2d_cuda(u, unew):
    M, N = u.shape     
    i, j = numba.cuda.grid(2)
    if i>=1 and i < M-1 and j >=1 and j < N-1 :
        unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )   
 

@numba.cuda.jit()
def lap2d_cuda2(u, unew):
    M, N = u.shape     
    i = numba.cuda.threadIdx.x + numba.cuda.blockIdx.x * numba.cuda.blockDim.x
    j = numba.cuda.threadIdx.y + numba.cuda.blockIdx.y * numba.cuda.blockDim.y

    if i>=1 and i < M-1 and j >=1 and j < N-1 :
        unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )


# Benchmark

M = 4096
N = 4096

u = np.zeros((M, N), dtype=np.float64)
unew = np.zeros((M, N), dtype=np.float64)

%timeit lap2d_cuda[(16,16),(16,16)](u, unew);numba.cuda.synchronize()
