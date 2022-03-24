N = 500
A = np.random.rand(N,N)
B = np.random.rand(N,N)

%timeit matmul_gu(A, B, C); numba.cuda.synchronize()
