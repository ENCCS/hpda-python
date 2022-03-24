@numba.cuda.jit
def math_kernel(a, b, result): # numba.cuda.jit does not return result yet
    pos = numba.cuda.grid(1)
    if (pos < a.shape[0]) and (pos < b.shape[0]):
        result[pos] = math.exp(a[pos]*b[pos])



a = np.random.rand(10000000)
b = np.random.rand(10000000)
c = np.random.rand(10000000)


threadsperblock = 32
blockspergrid = (10000000 + 31) // 32 # blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock

%timeit math_kernel[threadsperblock, blockspergrid](a, b, c)


@numba.cuda.jit
def math_kernel(a, b, result): # numba.cuda.jit does not return result yet
    pos = numba.cuda.grid(1)
    if (pos < a.shape[0]) and (pos < b.shape[0]):
        result[pos] = math.exp(a[pos]*b[pos])
    

@numba.cuda.jit(device=True)
def math_device(a, b):
    a = math.exp(a*b)
    return a


@numba.cuda.jit
def math_kernal_devicefunction(a, b, result): # cuda.jit does not return result yet
    pos = numba.cuda.grid(1)
    if (pos < a.shape[0]) and (pos < b.shape[0]):
        result[pos] = math_device(a[pos], b[pos])
   
    
