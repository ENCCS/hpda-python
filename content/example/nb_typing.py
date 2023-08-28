import numpy as np
import numba

@numba.jit(numba.float64[:](numba.float64[:]))
def nb_typing(arr):
   res=np.empty(len(arr))
   for i in range(len(arr)):
       res[i]=arr[i]**2
       
   return res
