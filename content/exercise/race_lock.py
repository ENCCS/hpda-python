from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Value, Lock

lock = Lock()

# adding lock
def inc(i):
    lock.acquire()
    val.value += 1
    lock.release()


# define a large number
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
