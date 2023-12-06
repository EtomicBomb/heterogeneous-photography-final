import numpy as np
from numpy import ctypeslib
import ctypes
import os

_gpu = ctypeslib.load_library(os.environ['SHARED_OBJECT_PATH'], '.')

_gpu.scanline_stereo.restype = ctypes.c_double
_gpu.scanline_stereo.argtypes = [
    ctypes.c_long, # rows
    ctypes.c_long, # cols_src
    ctypes.c_long, # cols_dst
    ctypes.c_long, # patch size
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C')), # src
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C')), # dst
    ctypeslib.ndpointer(dtype=np.int_, flags=('A', 'C', 'W')), # correspondance
    ctypeslib.ndpointer(dtype=np.byte, flags=('A', 'C', 'W')), # occlusion
]

def scanline_stereo(src, dst, patch_size):
    rows, cols_src = np.shape(src)
    rows_dst, cols_dst = np.shape(dst)
    assert rows == rows_dst
    src = np.require(src, dtype=np.float64, requirements=('A', 'C'))
    dst = np.require(dst, dtype=np.float64, requirements=('A', 'C'))
    correspondance = np.zeros((rows, cols_src))
    occlusion = np.zeros((rows, cols_src))
    result = _gpu.scanline_stereo(rows, cols_src, cols_dst, patch_size, src, dst, correspondance, occlusion)
    assert not np.isnan(result)
    return correspondance, occlusion

_gpu.matrix_vector.restype = ctypes.c_double
_gpu.matrix_vector.argtypes = [
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_long,
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C')),
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C')),
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C', 'W')),
]

def matrix_vector(matrix, vector):
    problem_count, rows, cols = np.shape(matrix)
    assert np.shape(vector) == (problem_count, cols, 1)
    matrix = np.require(matrix, dtype=np.float64, requirements=('A', 'C'))
    vector = np.require(vector, dtype=np.float64, requirements=('A', 'C'))
    result = np.zeros((problem_count, rows, 1))
    elapsed = _gpu.matrix_vector(problem_count, rows, cols, matrix, vector, result)
    if np.isnan(elapsed):
        elapsed = None
    return result, elapsed

