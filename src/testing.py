import sys
import numpy as np
from skimage import data, transform, util, color, measure, feature, registration
from skimage import io as skimage_io
import io
import time
from scipy import io as scipy_io

from bindings import *


def performance_statistics():
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
            print(patch_size, rows_small, cols_src_small, cols_dst_small, ' '.join(map(str, np.median(all_timings, axis=0))))
