import dask
import time
import numpy as np
from dask.distributed import Client, LocalCluster

def calc_mean(i, n):
    data = np.mean(np.random.normal(size = n))
    return(data)
    
n = 100000
output = [dask.delayed(calc_mean)(i, n) for i in range(100)]

cluster = LocalCluster(n_workers = 1,threads_per_worker=1)
c = Client(cluster)

%timeit dis_1 = dask.compute(output,n_workers = 1)
#619 ms ± 253 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

cluster.scale(2)
%timeit dis_2 = dask.compute(output,n_workers = 2)
#357 ms ± 131 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

cluster.scale(4)
%timeit dis_4 = dask.compute(output,n_workers = 4)
#265 ms ± 53.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

c.shutdown()
