import pytest
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io
import time

from bindings import *

src = skimage_io.imread('inputs/Bowling1/view1.png')
dst = skimage_io.imread('inputs/Bowling1/view5.png')
src = util.img_as_float64(src)
dst = util.img_as_float64(dst)
src = color.rgb2gray(src)
dst = color.rgb2gray(dst)

rows, cols_src = np.shape(src)
_, cols_dst = np.shape(dst)

trial_count = 20
occlusion_cost = 0.001
num_threads = 8

all_timings = []
for patch_size in np.int_(np.geomspace(1, 30, 4)):
    for trial in range(trial_count):
#         correspondence, valid, timings = scanline_stereo_gpu(src, dst, patch_size, occlusion_cost)
        correspondence, valid, timings = scanline_stereo_cpu_naive(src, dst, patch_size, occlusion_cost, num_threads)
        all_timings.append(timings)
    print(patch_size, rows, cols_src, cols_dst, ' '.join(map(str, np.median(all_timings, axis=0))))
