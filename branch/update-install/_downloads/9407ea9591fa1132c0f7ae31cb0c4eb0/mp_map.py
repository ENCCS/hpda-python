import multiprocessing as mp
   
def square(x):
    return x * x
   
if __name__ == '__main__':
    nprocs = mp.cpu_count()
    print(f"Number of CPU cores: {nprocs}")
   
    # use context manager to allocate and release the resources automatically
    with mp.Pool(processes=nprocs) as pool:
        result = pool.map(square, range(20))    
    print(result)
    
