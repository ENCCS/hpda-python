from numba import jit

@jit('f8(f8[:])')
def f_numba_dtype(x):
    return x ** 2 - x

@jit('f8(f8,f8,i4)')
def integrate_f_numba_dtype(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_numba_dtype(a + i * dx)
    return s * dx

@jit('f8(f8[:],f8[:],i4[:])')
def apply_integrate_f_numba_dtype(col_a, col_b, col_N):
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f_numba_dtype(col_a[i], col_b[i], col_N[i])

    return res

%timeit apply_integrate_f_numba_dtype(df['a'],df['b'],df['N'])
%timeit apply_integrate_f_numba_dtype(np.asarray(df['a']),np.asarray(df['b']),np.asarray(df['N'],dtype=np.int32))



from numba import float64, njit
@nb.jit
def mean_distance(x, y):
    nx = len(x)
    result = 0.0
    count = 0
    for i in range(nx):
        result += x[i] - y[i]
        count += 1
    return result / count

@njit(float64(float64[:], float64[:]))
def mean_distance2(x, y):
    return (x - y).mean()

x = np.random.randn(10000000)
y = np.random.randn(10000000)


import numpy as np

def smooth(x):
    out = np.empty_like(x)
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            out[i, j] = (x[i + -1, j + -1] + x[i + -1, j + 0] + x[i + -1, j + 1] +
                         x[i +  0, j + -1] + x[i +  0, j + 0] + x[i +  0, j + 1] +
                         x[i +  1, j + -1] + x[i +  1, j + 0] + x[i +  1, j + 1]) // 9

    return out

import numba

@numba.njit
def numba_smooth(x):
    out = np.empty_like(x)
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            out[i, j] = (x[i + -1, j + -1] + x[i + -1, j + 0] + x[i + -1, j + 1] +
                         x[i +  0, j + -1] + x[i +  0, j + 0] + x[i +  0, j + 1] +
                         x[i +  1, j + -1] + x[i +  1, j + 0] + x[i +  1, j + 1]) // 9

    return out


import numba

@numba.njit('f8(f8[:,:])')
def numba_dtype_smooth(x):
    out = np.empty_like(x)
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            out[i, j] = (x[i + -1, j + -1] + x[i + -1, j + 0] + x[i + -1, j + 1] +
                         x[i +  0, j + -1] + x[i +  0, j + 0] + x[i +  0, j + 1] +
                         x[i +  1, j + -1] + x[i +  1, j + 0] + x[i +  1, j + 1]) // 9

    return out
