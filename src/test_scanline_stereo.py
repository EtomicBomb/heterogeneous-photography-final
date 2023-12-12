import pytest
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io

from bindings import *

def image_pair():
    src = skimage_io.imread('inputs/Bowling1/view1.png')
    dst = skimage_io.imread('inputs/Bowling1/view5.png')
    src = util.img_as_float64(src)
    dst = util.img_as_float64(dst)
    src = color.rgb2gray(src)
    dst = color.rgb2gray(dst)
    return src, dst

def test_one():
    src, dst = image_pair()
    patch_size = 10
    occlusion_cost = 0.01
    num_cpu_threads = 32
    correspondence0, valid0, gpu_timings = scanline_stereo_gpu(src, dst, patch_size, occlusion_cost)
    correspondence1, valid1, _ = scanline_stereo_naive_gpu(src, dst, patch_size, occlusion_cost)
    correspondence2, valid2, cpu_timings = scanline_stereo_cpu(src, dst, patch_size, occlusion_cost, num_cpu_threads)
    print('cpu timings', cpu_timings)
    print('gpu timings', gpu_timings)
    assert np.allclose(valid0, valid1)
    assert np.allclose(valid0, valid2)
    assert np.allclose(correspondence0, correspondence1)
    assert np.allclose(correspondence0, correspondence2)

# TODO: these would be helpful if we wanted to test the intermediate results
# def cum_sum_diagonal(expected):
#     rows, src_cols, dst_cols = np.shape(expected)
#     for r in range(rows):
#         for s0 in range(src_cols):
#             s = s0 + 1
#             d = 1
#             while s < src_cols and d < dst_cols:
#                 expected[r, s, d] += expected[r, s-1, d-1]
#                 s += 1
#                 d += 1
#         for d0 in range(1, dst_cols):
#             s = 1
#             d = d0 + 1
#             while s < src_cols and d < dst_cols:
#                 expected[r, s, d] += expected[r, s-1, d-1]
#                 s += 1
#                 d += 1
#     return expected
# 
# def pixel_similarity(src, dst):
#     rows, cols_src = np.shape(src)
#     _, cols_dst = np.shape(dst)
#     src = np.reshape(src, (rows, cols_src, 1))
#     dst = np.reshape(dst, (rows, 1, cols_dst))
#     expected = np.square(src - dst)
#     expected = np.cumsum(expected, axis=0)
#     expected = cum_sum_diagonal(expected)
#     return expected

