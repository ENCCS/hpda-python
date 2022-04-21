%%cython

import numpy as np
import pandas as pd

cdef double f_cython_dtype2(double x):
    return x ** 2 - x

cpdef double integrate_f_cython_dtype2(double a, double b, long N):   
    cdef double s, dx
    cdef long i
    
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cython_dtype2(a + i * dx)
    return s * dx

cpdef double[:] apply_integrate_f_cython_dtype2(double[:] col_a, double[:] col_b, long[:] col_N):
    cdef long n,i
    cdef double[:] res
    
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_dtype2(col_a[i], col_b[i], col_N[i])
    return res
