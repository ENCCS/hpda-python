import numpy as np
import numba
import numba.cuda
import time

@numba.cuda.jit
def matmul_kernel(A, B, C):
    i, j = numba.cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# Benchmark

# first generate double precision input data

N = 8192
A = np.random.rand(N,N)
B = np.random.rand(N,N)
C = np.random.rand(N,N)

# copy them to GPU

d_A = numba.cuda.to_device(A)
d_B = numba.cuda.to_device(B)
d_C = numba.cuda.to_device(C)

# setup grid and block

threadsperblock = (16, 16)
blockspergrid = (10,10)

# create array to save profiling information
n_loop=20
test1=np.empty(n_loop)
test2=np.empty(n_loop)


# benchmark double precision input data

for i in range(n_loop):
    t_s=time.time()
    matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C); numba.cuda.synchronize()
    t_e=time.time()
    test1[i]=t_e - t_s

# then generate single precision input data

d_A32 = numba.cuda.to_device(A.astype(np.float32))
d_B32 = numba.cuda.to_device(B.astype(np.float32))
d_C32 = numba.cuda.to_device(C.astype(np.float32))

# benchmark single precision input data

for i in range(n_loop):
    t_s=time.time()
    matmul_kernel[blockspergrid, threadsperblock](d_A32, d_B32, d_C32); numba.cuda.synchronize()
    t_e=time.time()
    test2[i]=t_e - t_s


# calculate mean runtime

record = test1
print("matmul_kernel dtype64 Runtime")
print("average {:.5f} second (except 1st run)".format(record[1:].mean()))

record = test2
print("matmul_kernel dtype32 Runtime")
print("average {:.5f} second (except 1st run)".format(record[1:].mean()))
