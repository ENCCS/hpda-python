import math
import concurrent.futures
import time

# Grid size
n = 100000000
# Number of threads
numthreads = 4

def integration_concurrent(threadindex, n=n, numthreads=numthreads):
    h = 1.0 / float(n)  
    mysum = 0.0
    workload = n/numthreads
    begin = int(workload*threadindex)
    end = int(workload*(threadindex+1))
    
    for i in range(begin, end):
        x = h * (i + 0.5)
        mysum += x ** (3/2)

    return h * mysum

if __name__ == "__main__":
    print(f"Using {numthreads} threads")
    starttime = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        partial_integrals = list(executor.map(integration_concurrent, range(numthreads)))

    integral = sum(partial_integrals)
    endtime = time.time()

    print("Integral value is %e, Error is %e" % (integral, abs(integral - 2/5)))  # The correct integral value is 2/5
    print("Time spent: %.2f sec" % (endtime-starttime))


# 50.17 sec 
