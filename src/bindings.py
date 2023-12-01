import numpy as np
from numpy import ctypeslib
import ctypes
import os

_gpu = ctypeslib.load_library(os.environ['SHARED_OBJECT_PATH'], '.')

_gpu.matrix_vector.restype = ctypes.c_double
_gpu.matrix_vector.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypeslib.ndpointer(dtype=np.float32, flags=('A', 'C')),
    ctypeslib.ndpointer(dtype=np.float32, flags=('A', 'C')),
    ctypeslib.ndpointer(dtype=np.float32, flags=('A', 'C', 'W')),
]

def matrix_vector(matrix, vector):
    problem_count, rows, cols = np.shape(matrix)
    assert np.shape(vector) == (problem_count, cols, 1)
    matrix = np.require(matrix, dtype=np.float32, requirements=('A', 'C'))
    vector = np.require(vector, dtype=np.float32, requirements=('A', 'C'))
    result = np.zeros((problem_count, rows, 1))
    elapsed = _gpu.matrix_vector(problem_count, rows, cols, matrix, vector, result)
    if np.isnan(elapsed):
        elapsed = None
    return result, elapsed

