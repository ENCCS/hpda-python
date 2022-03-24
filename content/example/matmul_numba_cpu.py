@numba.guvectorize(['(float64[:,:], float64[:,:], float64[:,:])'], '(m,l),(l,n)->(m,n)', target='cpu')
def matmul_numba_cpu(A,B,C):
    a = A.shape[0]
    b = B.shape[1]
    c = B.shape[0]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                C[i,j] += A[i,k] * B[k,j]
