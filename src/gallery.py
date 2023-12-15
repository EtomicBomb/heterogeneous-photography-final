import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from matplotlib.gridspec import GridSpec
from skimage import data, transform, util, color, measure, feature, registration
import imageio.v3 as iio
import io
import time
from pathlib import Path

from bindings import *

def plot_all(images):
    fig, axs = plt.subplots(len(images), 3)
    for ax in np.ravel(axs):
        ax.axis('off')
    for (src, dst, disp, correspondence), (ax1, ax2, ax3) in zip(images, axs):
        ax1.imshow(src)
        ax2.imshow(disp, cmap='gray')
        ax3.imshow(correspondence, cmap='gray', vmin='20', vmax='120')
    plt.show()


images = []
sources = Path() / 'inputs' / 'half'
for source in sorted(list(sources.iterdir())):
    src = source / 'view1.png'
    dst = source / 'view5.png'
    disp = source / 'disp1.png'
    src = iio.imread(src)
    dst = iio.imread(dst)
    disp = iio.imread(disp)
    src_gray = color.rgb2gray(src)
    dst_gray = color.rgb2gray(dst)
    occlusion_cost = 10 ** -3.0
    patch_size = 8
    num_threads = 8
    correspondence, valid, timings = scanline_stereo_cpu(src_gray, dst_gray, patch_size, occlusion_cost, num_threads)
    correspondence = np.arange(src_gray.shape[1]) - correspondence
    correspondence = correspondence * valid
    images.append((src, dst, disp, correspondence))
    print('ran')

plot_all(images[:9])
plot_all(images[9:])

