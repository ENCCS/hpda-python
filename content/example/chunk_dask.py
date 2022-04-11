import dask
import dask.array as da

%%time
x = da.random.random((20000, 20000), chunks=(1000, 1000))
y = x.mean(axis=0)
y.compute()
