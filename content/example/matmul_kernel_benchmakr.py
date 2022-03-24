N = 500
A = np.random.rand(N,N)
B = np.random.rand(N,N)

TPB = 16
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

%timeit matmul_kernel[blockspergrid, threadsperblock](A, B, C); numba.cuda.synchronize()
