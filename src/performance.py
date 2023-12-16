import pytest
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io
import time

from bindings import *

src = skimage_io.imread('inputs/half/Bowling1/view1.png')
dst = skimage_io.imread('inputs/half/Bowling1/view5.png')
src = util.img_as_float64(src)
dst = util.img_as_float64(dst)
src = color.rgb2gray(src)
dst = color.rgb2gray(dst)

rows, cols_src = np.shape(src)
_, cols_dst = np.shape(dst)

trial_count = 20
occlusion_cost = 0.001
num_threads = 8
patch_size = 12

all_timings = []
for size in [0.25, 0.5, 0.75, 1.0]:
    rows_small, cols_src_small, cols_dst_small = np.int_(np.round(np.multiply(size, [rows, cols_src, cols_dst])))
    src_small = transform.resize(src, (rows_small, cols_src_small), anti_aliasing=True)
    dst_small = transform.resize(dst, (rows_small, cols_dst_small), anti_aliasing=True)

# for patch_size in np.int_(np.round(np.geomspace(1, 30, 4))):
#     src_small = src
#     dst_small = dst
    print('hello world')
    start = time.time()
    for trial in range(trial_count):
        correspondence, valid, timings = scanline_stereo_gpu(src_small, dst_small, patch_size, occlusion_cost)
#         correspondence, valid, timings = scanline_stereo_cpu(src_small, dst_small, patch_size, occlusion_cost, num_threads)
        all_timings.append(timings)
    print((time.time() - start) / trial_count)
    print(rows_small, ' '.join(map(str, np.median(all_timings, axis=0))))
