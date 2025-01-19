from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Value

# define a function to increment the value by 1
def inc(i):
    val.value += 1

# using a large number to see the problem
n = 100000

# create a shared data and initialize it to 0
val = Value('i', 0)
with ThreadPoolExecutor(max_workers=4) as pool:
    pool.map(inc, range(n))

print(val.value)

# create a shared data and initialize it to 0
val = Value('i', 0)
with ProcessPoolExecutor(max_workers=4) as pool:
    pool.map(inc, range(n))

print(val.value)
