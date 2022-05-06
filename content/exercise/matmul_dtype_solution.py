import numpy as np
import numba
import numba.cuda

@numba.cuda.jit
def matmul_kernel_float32(A, B, C):
    i, j = numba.cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = numba.float32(0.)
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp
