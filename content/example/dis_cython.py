%load_ext Cython # load Cython in Jupyter Notebook
%%cython
cimport cython
cimport numpy as np
import numpy as np

def dis_cython(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D



%%cython
cimport cython
cimport numpy as np
import numpy as np

cpdef double[:,:] dis_cython_dtype(double[:,:] X):
    cdef int i,j,k,M,N
    cdef double d,tmp
    
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D



%%cython
cimport cython
cimport numpy as np
import numpy as np

cpdef np.ndarray[double, ndim=2] dis_cython_dtype2(np.ndarray[double,ndim=2] X):
    cdef i,j,k,M,N
    cdef double d,tmp
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.double)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D







%%cython
cimport cython
cimport numpy as np
import numpy as np

cpdef double[:,:] dis_cython_dtype_gs(double[:,:] X):
    cdef Py_ssize_t i,j,k
    cdef double d,tmp
    cdef Py_ssize_t M = X.shape[0]
    cdef Py_ssize_t N = X.shape[1]
    D = np.zeros((M, M), dtype=np.float64)

    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D


X = np.random.random((1000, 3))
%timeit dis_cython(X)







X = np.random.random((1000, 3))
%timeit dis_cython(X)
