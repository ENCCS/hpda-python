%%cython
import numpy as np

def f_cython(x):
    return x * (x - 1)

def integrate_f_cython(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
    	s += f_cython(a + i * dx)
    return s * dx

def apply_integrate_f_cython(col_a, col_b, col_N):
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f_cython(col_a[i], col_b[i], col_N[i])

    return res


%timeit apply_integrate_f_cython(df['a'], df['b'], df['N'])
