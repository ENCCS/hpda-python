from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math

def func_cpu_bound(n):
    sum = 0
    for i in range(1, n+1):
        sum += math.exp(math.log(math.sqrt(math.pow(i, 3.0))))
    return sum


n=1000000

%%time
func_cpu_bound(n)

%%time
with ThreadPoolExecutor(max_workers=4) as pool:
    pool.map(func_cpu_bound, [n,n,n,n])

%%time
with ProcessPoolExecutor(max_workers=4) as pool:
    pool.map(func_cpu_bound, [n,n,n,n])
