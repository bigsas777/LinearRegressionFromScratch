import numpy as np

def add_ones(input_matrix):
    """
    Adds a column of ones to the input matrix to account for the theta0 parameter.

    Args:
        input_matrix (numpy.ndarray): Input matrix with dimensions (m, n).

    Returns:
        numpy.ndarray: Input matrix with an added column of ones, dimensions (m, n+1).
    """
    m = input_matrix.shape[0]
    ones = np.ones((m, 1))  # Column of ones to handle theta0
    input_matrix = np.hstack((ones, input_matrix))
    return input_matrix

def ensure_2d(array):
    """
    Ensures that the array is two-dimensional.

    Args:
        array (numpy.ndarray): Array to check.

    Returns:
        numpy.ndarray: Two-dimensional array.
    """
    return array if array.ndim > 1 else array.reshape(-1, 1)