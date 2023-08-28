import numpy as np
import numba

@numba.jit(numba.float64[::1](numba.float64[::1]))
def nb_typing_vec(arr):
   res=np.empty(len(arr))
   for i in range(len(arr)):
       res[i]=arr[i]**2
    
   return res
