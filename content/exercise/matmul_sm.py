import numba

TPB = 16
@numba.cuda.jit
def matmul_sm(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = numba.cuda.shared.array(shape=(TPB, TPB), dtype=numba.float64)
    sB = numba.cuda.shared.array(shape=(TPB, TPB), dtype=numba.float64)

    x, y = numba.cuda.grid(2)

    tx = numba.cuda.threadIdx.x
    ty = numba.cuda.threadIdx.y
    bpg = numba.cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        # Wait until all threads finish preloading
        numba.cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        numba.cuda.syncthreads()

    C[x, y] = tmp
