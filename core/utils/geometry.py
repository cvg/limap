import numpy as np

def to_homogeneous(arr):
    # Adds a new column with ones
    return np.hstack([arr, np.ones((len(arr), 1))])

def to_homogeneous_t(arr):
    # Adds a new row with ones
    return np.vstack([arr, np.ones((1, arr.shape[1]))])

def to_cartesian(arr):
    return arr[..., :-1] / arr[..., -1].reshape((-1,) + (1,) * (arr.ndim - 1))

def to_cartesian_t(arr):
    return arr[:-1] / arr[-1]

