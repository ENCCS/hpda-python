@numba.cuda.jit(device=True)
def math_device(a, b):
    a = math.exp(a*b)
    return a

@numba.cuda.jit
def math_kernel_devicefunction(a, b, result): # cuda.jit does not return result yet
    pos = numba.cuda.grid(1)
    if (pos < a.shape[0]) and (pos < b.shape[0]):
        result[pos] = math_device(a[pos], b[pos])
   
    
