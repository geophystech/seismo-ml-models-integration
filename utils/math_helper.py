import math
import numpy as np


def normal_d(size = 20,  k = 0.95):
    """
    Returns normal distribution NumPy array.
    """

    default_size = 15.
    f_size = float(size)

    x = np.linspace(int(-size / 2), int(size / 2), size)

    # Calculate ND
    u = 0
    m = default_size / f_size
    q = np.sqrt(5.)

    return (k * 6 / (q * np.sqrt(2 * math.pi))) * np.power(math.e, -0.5 * ((x * m - u)/q)**2)


def insert_np(n_array, target, start = 0):
    """
    Inserts NumPy 1d array into target 1d array from starting position, treats out of bounds correctly.
    """
    if start > target.shape[0]:
        return

    # Target start/end
    start_t = start
    end_t = start + n_array.shape[0]

    start_i = 0
    if start < 0:
        start_i = -start
        start_t = 0

    if start_i > n_array.shape[0]:
        return

    end_i = n_array.shape[0]
    if end_t > target.shape[0]:
        end_i = n_array.shape[0] - (end_t - target.shape[0])
        end_t = target.shape[0]

    if end_i <= 0:
        return

    target[start_t : end_t] = n_array[start_i : end_i]

    return target
