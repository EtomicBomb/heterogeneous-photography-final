import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from matplotlib.gridspec import GridSpec
from skimage import data, transform, util, color, measure, feature, registration
import cv2
import imageio.v3 as iio
import io
import time
from pathlib import Path
import itertools

from bindings import *
from demo import *

class Main:
    def stream_next(self):
        src, dst = next(self.stream)
        src = self.corrupt_src(src)
        dst = self.corrupt_dst(dst)
        return src, dst

    def __init__(self):
        fig = plt.figure(num='demo', layout='constrained')
        gs = GridSpec(12, 3, figure=fig)

        # widgets
        ax = fig.add_subplot(gs[0, -1])
        ax = widgets.Slider(ax, 'patch size', 1, 50, valinit=5)
        self.patch_size = ax

        ax = fig.add_subplot(gs[1, -1])
        ax = widgets.Slider(ax, 'occlusion cost', -8, 1, valinit=-3.6)
        self.occlusion_cost = ax

        ax = fig.add_subplot(gs[2, -1])
        ax = widgets.Slider(ax, 'darken', 0, 2, valinit=1.0)
        ax.on_changed(self.corrupt_slider_update)
        self.darken = ax

        ax = fig.add_subplot(gs[3, -1])
        ax = widgets.Slider(ax, 'tint', -50, 50, valinit=0)
        ax.on_changed(self.corrupt_slider_update)
        self.tint = ax

        ax = fig.add_subplot(gs[4, -1])
        ax = widgets.Slider(ax, 'temperature', -50, 50, valinit=0)
        ax.on_changed(self.corrupt_slider_update)
        self.temperature = ax

        ax = fig.add_subplot(gs[5, -1])
        ax = widgets.Slider(ax, 'noise scale', 0, 25, valinit=0)
        ax.on_changed(self.corrupt_slider_update)
        self.noise_scale = ax

        ax = fig.add_subplot(gs[6, -1])
        ax = widgets.Button(ax, 'calibrate')
        ax.on_clicked(self.calibrate_clicked)
        self.calibrate = ax

        # images
        self.stream = zip(iio.imread('inputs/brian-left-small.mp4'), iio.imread('inputs/brian-right-small.mp4'))
        self.corrupt_src = self.corrupt_dst = lambda x: x
        first_src, first_dst = self.stream_next()
        shape_src = np.shape(first_src)
        shape_dst = np.shape(first_dst)
        self.rows, self.cols_src, _ = shape_src
        _, self.cols_dst, _ = shape_dst
        self.h1 = self.h2 = transform.ProjectiveTransform()

        ax = fig.add_subplot(gs[0:3, 0])
        ax.set_axis_off()
        ax.set(title='raw src')
        self.src_widget = ax.imshow(np.zeros(shape_src))

        ax = fig.add_subplot(gs[0:3, 1])
        ax.set_axis_off()
        ax.set(title='raw dst')
        self.dst_widget = ax.imshow(np.zeros(shape_dst))

        ax = fig.add_subplot(gs[3:6, 0])
        ax.set_axis_off()
        ax.set(title='rectified src')
        self.src_rectified_widget = ax.imshow(np.zeros(shape_src))

        ax = fig.add_subplot(gs[3:6, 1])
        ax.set_axis_off()
        ax.set(title='rectified dst')
        self.dst_rectified_widget = ax.imshow(np.zeros(shape_dst))

        ax = fig.add_subplot(gs[6:9, 0])
        ax.set_axis_off()
        ax.set(title='corrected src')
        self.src_color_widget = ax.imshow(np.zeros(shape_src))

        ax = fig.add_subplot(gs[6:9, 1])
        ax.set_axis_off()
        ax.set(title='corrected dst')
        self.dst_color_widget = ax.imshow(np.zeros(shape_dst))

        ax = fig.add_subplot(gs[9:12, 0])
        ax.set_axis_off()
        ax.set(title='mapped')
        self.mapped_widget = ax.imshow(np.zeros(shape_src))

        ax = fig.add_subplot(gs[9:12, 1])
        ax.set_axis_off()
        ax.set(title='integrated')
        self.integrated_widget = ax.imshow(np.zeros(shape_src))

        ani = animation.FuncAnimation(fig, self.animate, cache_frame_data=False, blit=True, interval=20)
        plt.show()

    def animate(self, frame_count):
        raw_src, raw_dst = self.stream_next()

        src, dst = raw_src, raw_dst
    #     src = transform.warp(src, self.h1)
    #     dst = transform.warp(dst, self.h2)

        src_gray = color.rgb2gray(src)
        dst_gray = color.rgb2gray(dst)
        dst_gray = dst_gray - np.mean(dst_gray) + np.mean(src_gray)

        correspondence, valid, timings = scanline_stereo_cpu(
                dst_gray, src_gray, 
                patch_size=int(self.patch_size.val), occlusion_cost=10**self.occlusion_cost.val, num_threads=8)
        pretty = np.arange(self.cols_src) - correspondence
        valid = valid & (pretty >= 0)
        pretty = pretty * valid
        pretty = np.clip((pretty - 8) / 30, 0, 1)
        pretty = color.gray2rgb(pretty)

        coords_dst = np.mgrid[:self.rows, :self.cols_dst]
        coords_src = np.expand_dims(np.arange(self.rows), 1)
        coords_src = np.broadcast_to(coords_src, (self.rows, self.cols_src))
        coords_src = np.stack((coords_src, correspondence), axis=0) 

        coords_src_valid = coords_src.reshape(2, -1)[:, np.flatnonzero(valid)]
        coords_dst_valid = coords_dst.reshape(2, -1)[:, np.flatnonzero(valid)]

        corrected_dst = estimate_model(src, dst, coords_src_valid, coords_dst_valid)

        coords_src_valid = np.expand_dims([self.rows, self.cols_src], (1, 2))
        coords_src_valid = valid & (coords_src >= 0) & (coords_src < coords_src_valid)
        coords_src_valid = np.all(coords_src_valid, axis=0)
        coords_src_valid = np.expand_dims(coords_src_valid, 2)


#         colors_src = dst
        colors_src = util.img_as_float64(src)
        colors_src = colors_src[tuple(coords_src)]
        colors_src = np.where(coords_src_valid, colors_src, np.nan)

        integrated = np.nan_to_num(colors_src, nan=corrected_dst)
#         integrated = np.nanmean([corrected_dst, colors_src], axis=0)

        self.src_widget.set_data(util.img_as_ubyte(raw_src))
        self.dst_widget.set_data(util.img_as_ubyte(raw_dst))

        self.src_rectified_widget.set_data(util.img_as_ubyte(src))
        self.dst_rectified_widget.set_data(util.img_as_ubyte(dst))

        self.src_color_widget.set_data(util.img_as_ubyte(src))
        self.dst_color_widget.set_data(util.img_as_ubyte(corrected_dst))

        self.mapped_widget.set_data(util.img_as_ubyte(colors_src))
        self.integrated_widget.set_data(util.img_as_ubyte(integrated))

        return [self.src_widget, self.dst_widget, self.src_rectified_widget, self.dst_rectified_widget, self.src_color_widget, self.dst_color_widget, self.mapped_widget, self.integrated_widget]
            
    def corrupt_slider_update(self, arg):
        src, dst = self.stream_next()
        self.corrupt_src = corrupt(self.rows, self.cols_src, 1.0, 0, 0, 0, 1)
        self.corrupt_dst = corrupt(self.rows, self.cols_dst, darken=self.darken.val, tint=self.tint.val, temperature=self.temperature.val, noise_scale=self.noise_scale.val, noise_count=3)

    def calibrate_clicked(self, arg):
        raw_src, raw_dst = self.stream_next()
        self.h1, self.h2 = rectify(raw_src, raw_dst, 0.4)

Main()
