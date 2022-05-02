import math
import numba

@numba.cuda.jit
def math_kernel(x, y, result): # numba.cuda.jit does not return result yet
    pos = numba.cuda.grid(1)
    if (pos < x.shape[0]) and (pos < y.shape[0]):
        result[pos] = math.pow(x[pos],3.0) + 4*math.sin(y[pos])
