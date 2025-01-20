from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Array


# define a function to increment the value by 1
def inc(i):
    ind = mp.current_process().ident % 4
    arr[ind] += 1

# define a large number
n = 100000

# create a shared data and initialize it to 0
arr = Array('i', [0]*4)
with ProcessPoolExecutor(max_workers=4) as pool:
    pool.map(inc, range(n))

print(arr[:],sum(arr))
