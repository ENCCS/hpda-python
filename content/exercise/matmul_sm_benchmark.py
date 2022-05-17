import numpy as np
import numba
import numba.cuda
import time

N = 8192
A = np.random.rand(N,N)
B = np.random.rand(N,N)
C = np.random.rand(N,N)

d_A = numba.cuda.to_device(A)
d_B = numba.cuda.to_device(B)
d_C = numba.cuda.to_device(C)

threadsperblock = (16, 16)
blockspergrid = (10,10)

n_loop=20
test3=np.empty(n_loop)

for i in range(n_loop):
    t_s=time.time()
    matmul_sm[blockspergrid, threadsperblock](d_A, d_B, d_C); numba.cuda.synchronize()
    t_e=time.time()
    test3[i]=t_e - t_s


record = test3
print("matmul_sm Runtime")
print("average {:.5f} second (except 1st run)".format(record[1:].mean()))
