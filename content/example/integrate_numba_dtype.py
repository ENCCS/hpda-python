import numpy as np
import numba

@numba.jit(numba.float64(numba.float64))
def f_numba_dtype(x):
    return x ** 2 - x

@numba.jit(numba.float64(numba.float64,numba.float64,numba.int64))
def integrate_f_numba_dtype(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_numba_dtype(a + i * dx)
    return s * dx

@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:],numba.int64[:]))
def apply_integrate_f_numba_dtype(col_a, col_b, col_N):
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_numba_dtype(col_a[i], col_b[i], col_N[i])
    return res
