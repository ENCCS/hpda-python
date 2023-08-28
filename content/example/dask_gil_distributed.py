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

%timeit
rs = dask.compute(output,n_workers = 1)

cluster.scale(4)
%timeit
rs = dask.compute(output,n_workers = 4)

cluster.scale(8)
%timeit
rs = dask.compute(output,n_workers = 8)

c.shutdown()
