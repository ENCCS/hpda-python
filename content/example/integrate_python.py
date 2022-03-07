def f(x):
    return x ** 2 - x


def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx


def apply_integrate_f(col_a, col_b, col_N):
    n = len(col_N)
    res = np.empty(n)
    for i in range(n):
        res[i] = integrate_f(col_a[i], col_b[i], col_N[i])

    return res
