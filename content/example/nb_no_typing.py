import numpy as np
import numba

@numba.jit
def nb_no_typing(arr):
   res=np.empty(len(arr))
   for i in range(len(arr)):
       res[i]=arr[i]**2
       
   return res
