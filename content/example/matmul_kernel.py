@numba.cuda.jit
def matmul_kernel(A, B, C):
    i, j = numba.cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp




%timeit out = np.dot(A, B)


for i in range(np_loop):
    t_start = timeit.default_timer()
    out = matmul_jit(A, B)
    t_end = timeit.default_timer()
    nb_jit[i] = t_end - t_start


    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)

TPB = 16
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(out.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(out.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)


for i in range(cuda_loop):
    t_start = timeit.default_timer()
    matmul[blockspergrid, threadsperblock](d_A, d_B, out)
    t_end = timeit.default_timer()
    cuda_matmul[i] = t_end - t_start



for i in range(cuda_loop):
    t_start = timeit.default_timer()
    matmul[blockspergrid, threadsperblock](A, B, out)
    t_end = timeit.default_timer()
    cuda_matmul[i] = t_end - t_start



for i in range(cuda_loop):
    t_start = timeit.default_timer()
    out = matmul_gu3(A, B)
    t_end = timeit.default_timer()
    cuda_matmul_gu3[i] = t_end - t_start


@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cuda')
def matmul_gu(A, B, out):
    i, j = numba.cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        out[i, j] = tmp



for i in range(cuda_loop):
    t_start = timeit.default_timer()
    out = matmul_gu4(A, B)
    t_end = timeit.default_timer()
    cuda_matmul_gu3[i] = t_end - t_start
