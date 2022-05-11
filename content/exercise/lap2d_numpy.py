def lap2d_numpy(u, unew):
    unew[1:-1,1:-1]=0.25*(u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:])          


# Benchmark

import numpy as np

M = 4096
N = 4096

u = np.zeros((M, N), dtype=np.float64)
unew = np.zeros((M, N), dtype=np.float64)

%timeit lap2d_numpy(u, unew)
