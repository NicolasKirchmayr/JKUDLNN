import numpy as np


def to_one_hot(y, k=None):
    """
    Compute a one-hot encoding from a vector of integer labels.

    Parameters
    ----------
    y : (N, ) ndarray
        The zero-indexed integer labels to encode.
    k : int, optional
        The number of distinct labels in `y`.

    Returns
    -------
    one_hot : (N, k) ndarray
        The one-hot encoding of the labels.
    """
    y = np.asarray(y, dtype='int')
    n = len(y)
    if k is None:
        k = np.amax(y) + 1

    one_hot = np.zeros((n, k))
    one_hot[np.arange(n), y] = 1
    return one_hot
