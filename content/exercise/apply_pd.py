import numpy as np
import pandas as pd
import time

def linear_fit_loglog(row):
    X = np.log(np.arange(row.shape[0]) + 1.0)
    ones = np.ones(row.shape[0])
    A = np.vstack((X, ones)).T
    Y = np.log(row)
    res = np.linalg.lstsq(A, Y, rcond=-1)
    time.sleep(0.01)
    return res[0][0]


df = pd.read_csv("/ceph/hpc/home/euqiamgl/results.csv")
%timeit results = df.iloc[:,1:].apply(linear_fit_loglog, axis=1)
