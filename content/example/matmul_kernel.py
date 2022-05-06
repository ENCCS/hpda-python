import numba
import numba.cuda

@numba.cuda.jit
def matmul_kernel(A, B, C):
    i, j = numba.cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


@numba.cuda.jit
def matmul_kernel2(A, B, C):

    tx = numba.cuda.threadIdx.x
    ty = numba.cuda.threadIdx.y
    bx = numba.cuda.blockIdx.x
    by = numba.cuda.blockIdx.y
    bw = numba.cuda.blockDim.x
    bh = numba.cuda.blockDim.y

    i = tx + bx * bw
    j = ty + by * bh

    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp
