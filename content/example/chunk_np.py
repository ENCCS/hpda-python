import numpy as np

%%time
x = np.random.random((20000, 20000))
y = x.mean(axis=0)
