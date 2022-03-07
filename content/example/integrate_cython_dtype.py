%%cython
cimport cython
cimport numpy as np
import numpy as np

cdef double f_cython_dtype(double x):
    return x * (x - 1)

cpdef double integrate_f_cython_dtype(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cython_dtype(a + i * dx)
    return s * dx


cpdef np.ndarray[double] apply_integrate_f_cython_dtype(np.ndarray[double] col_a,
                                                np.ndarray[double] col_b,
                                                np.ndarray[int] col_N):
    cdef int i, n
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f_cython_dtype(col_a[i], col_b[i], col_N[i])

    return res


cpdef double[:] apply_integrate_f_cython_dtype2(double[:] col_a,
                                                double[:] col_b,
                                                int[:] col_N):
    cdef int i, n
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f_cython_dtype(col_a[i], col_b[i], col_N[i])

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] apply_integrate_f_cython_dtype3(double[:] col_a,
                                                double[:] col_b,
                                                int[:] col_N):
    cdef int i, n
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f_cython_dtype(col_a[i], col_b[i], col_N[i])

    return res






%timeit apply_integrate_f_cython_dtype(np.asarray(df['a']),np.asarray(df['b']),np.asarray(df['N'],dtype=np.int32))
