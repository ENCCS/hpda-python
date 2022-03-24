import math
import numba

@numba.vectorize(['float64(float64, float64)'], target='cpu')
def func_numba_cpu(a, b):
    return math.pow(a*b, 1./2)/math.exp(a*b/1000)
