import sys
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io
import time

from bindings import *

def cum_sum_diagonal(expected):
    rows, src_cols, dst_cols = np.shape(expected)
    for r in range(rows):
        for s0 in range(src_cols):
            s = s0 + 1
            d = 1
            while s < src_cols and d < dst_cols:
                expected[r, s, d] += expected[r, s-1, d-1]
                s += 1
                d += 1
        for d0 in range(1, dst_cols):
            s = 1
            d = d0 + 1
            while s < src_cols and d < dst_cols:
                expected[r, s, d] += expected[r, s-1, d-1]
                s += 1
                d += 1
    return expected

def pixel_similarity(src, dst):
    rows, cols_src = np.shape(src)
    _, cols_dst = np.shape(dst)
    src = np.reshape(src, (rows, cols_src, 1))
    dst = np.reshape(dst, (rows, 1, cols_dst))
    expected = np.square(src - dst)
    expected = np.cumsum(expected, axis=0)
    expected = cum_sum_diagonal(expected)
    return expected

src = skimage_io.imread('inputs/a.jpg')
dst = skimage_io.imread('inputs/a.jpg')
src = util.img_as_float64(src)
dst = util.img_as_float64(dst)
src = color.rgb2gray(src)
dst = color.rgb2gray(dst)
rows, cols_src = np.shape(src)
_, cols_dst = np.shape(dst)
patch_size = 15

correspondance, valid, result = scanline_stereo(src, dst, patch_size)
print(np.min(result), np.max(result))



# pixel_similarity = np.square(np.reshape(src, (rows, cols_src, 1)) - np.reshape(dst, (rows, 1, cols_dst)))
# 
# r, s, d = rows-1, cols_src-1, cols_dst - 1
# total = 0
# for r_prime in range(r+1):
#     for s_prime in range(s+1):
#         d_prime = s_prime + d - s
#         if d_prime >= 0 and d_prime < cols_dst:
#             print(s_prime, d_prime, pixel_similarity[r_prime, s_prime, d_prime])
#             total += pixel_similarity[r_prime, s_prime, d_prime]
# 
# print(total, result[r, s, d])




# correspondance, valid, result = scanline_stereo(src, dst, patch_size)
# expected = pixel_similarity(src, dst)
# print(np.sum(~np.isclose(result, expected)))
# assert np.allclose(result, expected)
