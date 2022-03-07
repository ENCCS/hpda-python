from numba import jit

@jit
def f_numba(x):
    return x ** 2 - x

@jit
def integrate_f_numba(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_numba(a + i * dx)
    return s * dx

@jit
def apply_integrate_f_numba(col_a, col_b, col_N):
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f_numba(col_a[i], col_b[i], col_N[i])

    return res

%timeit apply_integrate_f_numba(df['a'],df['b'],df['N'])



