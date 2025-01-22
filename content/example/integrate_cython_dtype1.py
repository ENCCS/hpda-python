%%cython -a

import numpy as np
import pandas as pd

cdef f_cython_dtype1(double x):
    return x ** 2 - x

cpdef integrate_f_cython_dtype1(double a, double b, long N):   
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cython_dtype1(a + i * dx)
    return s * dx

cpdef apply_integrate_f_cython_dtype1(double[:] col_a, double[:] col_b, long[:] col_N):
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_dtype1(col_a[i], col_b[i], col_N[i])
    return res
