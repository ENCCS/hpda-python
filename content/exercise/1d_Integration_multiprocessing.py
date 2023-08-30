import multiprocessing as mp
import math
import time

# Grid size
n = 100000000
nprocs = 4

def integration_process(pool_index, n, numprocesses):
    h = 1.0 / float(n)  
    mysum = 0.0
    workload = n / numprocesses

    begin = int(workload * pool_index)
    end = int(workload * (pool_index + 1))
    
    for i in range(begin, end):
        x = h * (i + 0.5)
        mysum += x ** (3/2)

    return h * mysum

if __name__ == '__main__':
    print(f"Using {nprocs} processes")

    starttime = time.time()

    with mp.Pool(processes=nprocs) as pool:
        partial_integrals = pool.starmap(integration_process, [(i, n, nprocs) for i in range(nprocs)])

    integral = sum(partial_integrals)
    endtime = time.time()

    print("Integral value is %e, Error is %e" % (integral, abs(integral - 2/5)))  # The correct integral value is 2/5
    print("Time spent: %.2f sec" % (endtime - starttime))

# 3.53 sec
