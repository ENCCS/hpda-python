import numba

@numba.cuda.jit()
def lap2d_cuda(u, unew):

    M = u.shape[0]     
    N = u.shape[1]


    i, j = numba.cuda.grid(2)
    if i>=1 and i < M-1 and j >=1 and j < N-1 :
        unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )   
 

@numba.cuda.jit()
def lap2d_cuda2(u, unew):

    M = u.shape[0]     
    N = u.shape[1]

    i = numba.cuda.threadIdx.x + numba.cuda.blockIdx.x * numba.cuda.blockDim.x
    j = numba.cuda.threadIdx.y + numba.cuda.blockIdx.y * numba.cuda.blockDim.y

    if i>=1 and i < M-1 and j >=1 and j < N-1 :
        unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )   
 
