import numpy as np

def dis_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))
