import dask
import time
import numpy as np

def calc_mean(i, n):
    data = np.mean(np.random.normal(size = n))
    return(data)
    
n = 100000

%%timeit
rs=[calc_mean(i, n) for i in range(100)]
