import numpy as np
import dask.dataframe as dd
from scipy.optimize import curve_fit
import time

def powerlaw(x, A, s):
    return A * np.power(x, s)

def fit_powerlaw(row):
    X = np.arange(row.shape[0]) + 1.0
    params, cov = curve_fit(f=powerlaw, xdata=X, ydata=row, p0=[100, -1], bounds=(-np.inf, np.inf))
    time.sleep(0.01)
    return params[1]

ddf = dd.read_csv("https://raw.githubusercontent.com/ENCCS/hpda-python/main/content/data/results.csv")
ddf4=ddf.repartition(npartitions=4)

# Note the optional argument ``meta`` which is recommended for dask dataframes. 
# It should contain an empty ``pandas.DataFrame`` or ``pandas.Series`` 
# that matches the dtypes and column names of the output, 
# or a dict of ``{name: dtype}`` or iterable of ``(name, dtype)``.

results = ddf4.iloc[:,1:].apply(fit_powerlaw, axis=1, meta=(None, "float64"))
%timeit results.compute()
results.visualize()

