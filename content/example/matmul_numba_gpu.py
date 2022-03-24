@numba.guvectorize(['(float32[:,:], float32[:,:], float32[:,:])'], '(m,l),(l,n)->(m,n)', target='cuda')
def matmul_numba_gpu(A,B,C):
    a = A.shape[0]
    b = B.shape[1]
    c = B.shape[0]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                C[i,j] += A[i,k] * B[k,j]
