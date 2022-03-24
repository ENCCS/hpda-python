import numpy as np
import math

@numba.vectorize([float64(float64, float64)], target='cuda') 
def f_gpu(x, y):
    return x**3 + 4*math.sin(y) 
