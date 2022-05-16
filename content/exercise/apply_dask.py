import numpy as np
import pandas as pd
import time
import dask.dataframe as dd

def linear_fit_loglog(row):
    X = np.log(np.arange(row.shape[0]) + 1.0)
    ones = np.ones(row.shape[0])
    A = np.vstack((X, ones)).T
    Y = np.log(row)
    res = np.linalg.lstsq(A, Y, rcond=-1)
    time.sleep(0.01)
    return res[0][0]


ddf = dd.read_csv("/ceph/hpc/home/euqiamgl/results.csv")
ddf4=ddf.repartition(npartitions=4)

# Note the additional argument ``meta`` which is required for dask dataframes. 
# It should contain an empty ``pandas.DataFrame`` or ``pandas.Series`` 
# that matches the dtypes and column names of the output, 
# or a dict of ``{name: dtype}`` or iterable of ``(name, dtype)``.

results = ddf4.iloc[:,1:].apply(linear_fit_loglog, axis=1, meta=(None, "float64"))
%timeit results.compute()
results.visualize()

