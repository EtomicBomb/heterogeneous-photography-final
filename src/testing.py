import sys
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io
import time

from bindings import *

src = skimage_io.imread('inputs/a.jpg')
dst = skimage_io.imread('inputs/a.jpg')

src = util.img_as_float64(src)
dst = util.img_as_float64(dst)

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

src = color.rgb2gray(src)
dst = color.rgb2gray(dst)
patch_size = 15

src = np.ones((250, 375))
correspondance, occlusion, result = scanline_stereo_testing(src, dst, patch_size)
expected = pixel_similarity(src, dst)
print(np.sum(~np.isclose(result, expected)))
assert np.allclose(result, expected)

# correspondance, occlusion, result = scanline_stereo_testing(src, dst, patch_size)
# # print('expected[0,8,0]', expected[0, 8, 0])
# # print('sum isclose', np.sum(np.isclose(expected.ravel(), result.ravel())))
# # print('index of failure', np.where(~np.isclose(expected.ravel() , result.ravel())))
# # print('expected', expected.ravel()[3000:3005])
# # print('results ', result.ravel()[3000:3005])
# 
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(correspondance)
# axs[1].imshow(occlusion)
# plt.show()
