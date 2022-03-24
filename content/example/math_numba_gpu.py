import math
import numba

@numba.vectorize(['float64(float64, float64)'], target='cuda')
def func_numba_gpu(a, b):
    return math.pow(a*b, 1./2)/math.exp(a*b/1000)



@numba.vectorize([numba.float32(numba.float64, numba.float64)], target='cuda')
def func_numba_gpu(a, b):
    return math.pow(a*b, 1./2)/math.exp(a*b/1000)
