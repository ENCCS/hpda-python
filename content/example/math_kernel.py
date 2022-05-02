import math
import numba

@numba.cuda.jit
def math_kernel(a, b, result): # numba.cuda.jit does not return result yet
    pos = numba.cuda.grid(1)
    if (pos < a.shape[0]) and (pos < b.shape[0]): # make sure no out of bound
        result[pos] = math.exp(a[pos]*b[pos])
