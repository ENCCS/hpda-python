import multiprocessing as mp

def power_n(x, n):
    return x ** n

if __name__ == '__main__':
    nprocs = mp.cpu_count()
    print(f"Number of CPU cores: {nprocs}")

    with mp.Pool(processes=nprocs) as pool:
        result = pool.starmap(power_n, [(x, 2) for x in range(20)])
    print(result)
    
