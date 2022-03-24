import numpy as np

M = 4096
N = 4096

u = np.zeros((M, N), dtype=np.float64)
unew = np.zeros((M, N), dtype=np.float64)

a = 0.5                     
dx = 0.01 
dy = 0.01 
dt = dx**2*dy**2 / ( 2*a*(dx**2+dy**2) )

%timeit lap2d(u, unew, a, dt, dx, dy)
