from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import sleep

def func_io_bound(n):
    sleep(n)
    return

%%time
with ThreadPoolExecutor(max_worker=4) as pool:
    pool.map(func_io_bound, [1,3,1,7])


%%time
with ProcessPoolExecutor(max_worker=4) as pool:
    pool.map(func_io_bound, [1,3,1,7])
