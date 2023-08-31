import dask
import time
import numpy as np

def calc_mean(i, n):
    data = np.mean(np.random.normal(size = n))
    return(data)
    
n = 100000
output = [dask.delayed(calc_mean)(i, n) for i in range(100)]

%%timeit
with dask.config.set(scheduler='processes',num_workers=1):
    mp_1 = dask.compute(output)
#990 ms ± 39.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
with dask.config.set(scheduler='processes',num_workers=2):
    mp_2 = dask.compute(output)
#881 ms ± 17.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
with dask.config.set(scheduler='processes',num_workers=4):
    mp_4 = dask.compute(output)
#836 ms ± 10.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
