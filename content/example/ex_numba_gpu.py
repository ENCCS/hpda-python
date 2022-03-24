import math
import numba

@numba.cuda.jit
def func(a, b):
  for i in range(100000):
    c = math.pow(a*b, 1./2)/math.exp(a*b/1000)
  return c
