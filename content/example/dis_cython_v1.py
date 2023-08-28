%%cython

import numpy as np
from libc.math cimport sqrt

def dis_cython_v1(double[:,:] X):
    cdef ssize_t i,j,k,M,N
    cdef double d,tmp
    cdef double[:,:] D
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = sqrt(d)
    return D
