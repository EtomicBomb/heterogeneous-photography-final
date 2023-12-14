import numpy as np
from numpy import ctypeslib
import ctypes
import os

scanline_stereo_argtypes = [
    ctypes.c_long, # rows
    ctypes.c_long, # cols_src
    ctypes.c_long, # cols_dst
    ctypes.c_long, # patch size
    ctypes.c_double, # occlusion cost
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C')), # src
    ctypeslib.ndpointer(dtype=np.float64, flags=('A', 'C')), # dst
    ctypeslib.ndpointer(dtype=np.int_, flags=('A', 'C', 'W')), # correspondance
    ctypeslib.ndpointer(dtype=np.byte, flags=('A', 'C', 'W')), # valid
    ctypeslib.ndpointer(dtype=np.float32, flags=('A', 'C', 'W')), # timings
]

try:
    _gpu = ctypeslib.load_library('target/gpu', '.')
    _gpu.scanline_stereo.restype = ctypes.c_int
    _gpu.scanline_stereo.argtypes = scanline_stereo_argtypes

    _gpu.scanline_stereo_naive.restype = ctypes.c_int
    _gpu.scanline_stereo_naive.argtypes = scanline_stereo_argtypes
except:
    pass

def scanline_stereo_gpu(src, dst, patch_size, occlusion_cost):
    rows, cols_src = np.shape(src)
    rows_dst, cols_dst = np.shape(dst)
    assert rows == rows_dst
    src = np.require(src, dtype=np.float64, requirements=('A', 'C'))
    dst = np.require(dst, dtype=np.float64, requirements=('A', 'C'))
    correspondance = np.zeros((rows, cols_src), dtype=np.int_)
    valid = np.zeros((rows, cols_src), dtype=np.byte)
    timings = np.zeros((7,), dtype=np.float32)
    ok = _gpu.scanline_stereo(rows, cols_src, cols_dst, patch_size, occlusion_cost, src, dst, correspondance, valid, timings)
    assert ok >= 0
    return correspondance, valid, timings

_cpu = ctypeslib.load_library('target/cpu', '.')
_cpu.scanline_stereo.restype = ctypes.c_int
_cpu.scanline_stereo.argtypes = [*scanline_stereo_argtypes, ctypes.c_int]

def scanline_stereo_cpu(src, dst, patch_size, occlusion_cost, num_threads):
    rows, cols_src = np.shape(src)
    rows_dst, cols_dst = np.shape(dst)
    assert rows == rows_dst
    src = np.require(src, dtype=np.float64, requirements=('A', 'C'))
    dst = np.require(dst, dtype=np.float64, requirements=('A', 'C'))
    correspondance = np.zeros((rows, cols_src), dtype=np.int_)
    valid = np.zeros((rows, cols_src), dtype=np.byte)
    timings = np.zeros((7,), dtype=np.float32)
    ok = _cpu.scanline_stereo(rows, cols_src, cols_dst, patch_size, occlusion_cost, src, dst, correspondance, valid, timings, num_threads)
    assert ok >= 0
    return correspondance, valid, timings
