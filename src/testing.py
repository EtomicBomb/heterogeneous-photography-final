import sys
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io
import time
from scipy import io as scipy_io

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

# src = skimage_io.imread('inputs/20231111_0001.jpg')
# dst = skimage_io.imread('inputs/20231111_0002.jpg')

src = skimage_io.imread('inputs/Bowling1/view1.png')
dst = skimage_io.imread('inputs/Bowling1/view5.png')
src = util.img_as_float64(src)
dst = util.img_as_float64(dst)
src = color.rgb2gray(src)
dst = color.rgb2gray(dst)

rows, cols_src = np.shape(src)
_, cols_dst = np.shape(dst)

data = {}
trial_count = 20
occlusion_cost = 0.001

times_called = 0
for shape in np.geomspace(0.1, 1.0, 5):
    all_timings = []
    rows_small, cols_src_small, cols_dst_small = np.int_(shape * np.array([rows, cols_src, cols_dst]))
    src_small = transform.resize(src, (rows_small, cols_src_small), anti_aliasing=True)
    dst_small = transform.resize(dst, (rows_small, cols_dst_small), anti_aliasing=True)
    for patch_size in np.int_(np.geomspace(1, 30, 4)):
        for trial in range(trial_count):
            correspondence, valid, timings = scanline_stereo(src_small, dst_small, patch_size, occlusion_cost)
            all_timings.append( np.diff(timings)[:8])
            times_called += 1
        print(patch_size, rows_small, cols_src_small, cols_dst_small, ' '.join(map(str, np.mean(all_timings, axis=0))))

print(data)

# print('min, max of result', np.min(result), np.max(result))
# scipy_io.savemat('out.mat', {'correspondence': correspondence, 'valid': valid})

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




# correspondence, valid, result = scanline_stereo(src, dst, patch_size)
# expected = pixel_similarity(src, dst)
# print(np.sum(~np.isclose(result, expected)))
# assert np.allclose(result, expected)
