import numpy as np
import numba
import numba.cuda
import time

@numba.guvectorize(['void(float64[:,:],float64[:,:])'],'(m,n)->(m,n)',target='cuda')
def lap2d_numba_gu_gpu(u, unew):
    M, N = u.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )


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



M = 4096
N = 4096

u = np.zeros((M, N), dtype=np.float64)
unew = np.zeros((M, N), dtype=np.float64)



n_loop=20
test1=np.empty(n_loop)
test2=np.empty(n_loop)
test3=np.empty(n_loop)

for i in range(n_loop):
    t_s=time.time()
    lap2d_numba_gu_gpu(u, unew)
    t_e=time.time()
    test1[i]=t_e - t_s


for i in range(n_loop):
    t_s=time.time()
    lap2d_cuda[(16,16),(16,16)](u, unew); numba.cuda.synchronize()
    t_e=time.time()
    test2[i]=t_e - t_s

for i in range(n_loop):
    t_s=time.time()
    d_u = numba.cuda.to_device(u)
    d_unew = numba.cuda.to_device(unew)
    lap2d_cuda[(16,16),(16,16)](d_u, d_unew); numba.cuda.synchronize()
    d_unew.copy_to_host(unew)
    t_e=time.time()
    test3[i]=t_e - t_s



record = test1
print("Numba gufunc Runtime")
print("average {:.5f} second (except 1st run)".format(record[1:].mean()))

record = test2
print("Numba CUDA without explicit data transfer Runtime")
print("average {:.5f} second (except 1st run)".format(record[1:].mean()))

record = test3
print("Numba CUDA with explicit data transfer Runtime")
print("average {:.5f} second (except 1st run)".format(record[1:].mean()))
