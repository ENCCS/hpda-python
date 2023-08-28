import dask
import time
import numpy as np

def calc_mean(i, n):
    data = np.mean(np.random.normal(size = n))
    return(data)
    
n = 100000
output = [dask.delayed(calc_mean)(i, n) for i in range(100)]

%%timeit
with dask.config.set(scheduler='threads',num_workers=1):
    rs = dask.compute(output)

%%timeit
with dask.config.set(scheduler='threads',num_workers=4):
    rs = dask.compute(output)

%%timeit
with dask.config.set(scheduler='threads',num_workers=8):
    rs = dask.compute(output)
