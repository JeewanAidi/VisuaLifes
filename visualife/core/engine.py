import numpy as np

def matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Shapes {A.shape} and {B.shape} are not aligned. "
            "Columns of A must equal rows of B."
        )
    return A @ B 


def elementwise_multiply(A, B):
    if A.shape != B.shape:
        raise ValueError(
            f"Shapes {A.shape} and {B.shape} must match for elementwise multiply."
        )
    return A * B 


def sum_array(A, axis=None):
    return np.sum(A, axis=axis, keepdims=True)
