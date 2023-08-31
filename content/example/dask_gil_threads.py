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
    mt_1 = dask.compute(output)
#395 ms ± 18.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
with dask.config.set(scheduler='threads',num_workers=2):
    mt_2 = dask.compute(output)
#1.28 s ± 1.46 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
with dask.config.set(scheduler='threads',num_workers=4):
    mt_4 = dask.compute(output)
#1.28 s ± 3.84 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
