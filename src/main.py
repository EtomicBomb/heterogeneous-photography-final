import sys
from dataclasses import dataclass
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from matplotlib import image as pli
from skimage import data, transform, util, color, measure, feature, registration
import cv2
import imageio.v3 as iio
import io
import time

import bindings

# 2 matrices of floats (images)
def foo(src, dst):
    src = iio.imread('inputs/a.jpg')
    dst = iio.imread('inputs/b.jpg')

    src = util.img_as_float32(src)
    dst = util.img_as_float32(dst)

    src_gray = color.rgb2gray(src)
    dst_gray = color.rgb2gray(dst)

    extractor_src = feature.SIFT()
    extractor_src.detect_and_extract(src_gray)
    extractor_dst = feature.SIFT()
    extractor_dst.detect_and_extract(dst_gray)

    matches = feature.match_descriptors(extractor_src.descriptors, extractor_dst.descriptors, cross_check=True)
    essential, inliers = measure.ransac((extractor_src.keypoints[matches[:, 0]], extractor_dst.keypoints[matches[:, 1]]),
                            transform.EssentialMatrixTransform, 
                            min_samples=8,
                            residual_threshold=1, 
                            max_trials=5000,
                            random_state=np.random.default_rng(seed=9))

    src_norm = src_gray - np.mean(src_gray)
    dst_norm = dst_gray - np.mean(dst_gray)

    rows, cols, _ = src.shape
    row, col = 15, 30
    x = [row, col, 1]
    x = np.reshape(x, (3, 1))
    ex = essential.params @ x

    src_patch = src_norm[row-10:row+10, col-10:col+10]

    rows, cols, _ = dst.shape
    indices = np.mgrid[:rows, :cols]
    indices = np.concatenate((indices, np.ones((1, rows, cols))), axis=0)
    weights = np.expand_dims(ex, 2) * indices
    weights = np.sum(weights, axis=0)
    sigma = 0.04
    weights = np.exp(-np.square(weights) / (2 * np.square(sigma)))
    
    correlation = scipy.signal.correlate2d(dst_norm, src_patch, mode='same')
    print(correlation.shape)

    print(weights.shape, correlation.shape, dst.shape)

    # x'Ex = 0 => x' [l1, l2, l3] = 0 => l1 x1 + l2 y1 + l3 = 0
    xs = np.arange(cols)
    l1, l2, l3 = ex
    ys = (-l3 - l1*xs)/l2
    valid = np.logical_and(ys >= 0, ys < rows)
    xs = np.int_(xs[valid])
    ys = np.int_(ys[valid])
    print(xs)
    print(ys)

    dst[ys, xs, :] = [255, 0, 0]
    fig, ax = plt.subplots(1, 5)
    ax[0].imshow(src)
    ax[1].imshow(dst)
    ax[2].imshow(weights)
    ax[3].imshow(correlation)
    ax[4].imshow(weights * correlation)

    plt.show()


def estimate_model(src, dst, coords_src, coords_dst):
    colors_src = src[coords_src[0], coords_src[1], :]
    colors_src = color.rgb2lab(colors_src)
    colors_dst = dst[coords_dst[0], coords_dst[1], :]
    colors_dst = color.rgb2lab(colors_dst)
    valid = np.all((colors_src[:,0] > 10, colors_src[:,0] < 90, colors_dst[:,0] > 10, colors_dst[:,0] < 90), axis=0)
    colors_src = colors_src[valid, :]
    colors_dst = colors_dst[valid, :]
    new_src = []
    src_lab = color.rgb2lab(src)
    for src_lab, channel_src, channel_dst in zip(np.moveaxis(src_lab, -1, 0), np.transpose(colors_src), np.transpose(colors_dst)):
        result = scipy.stats.linregress(channel_src, channel_dst)
        new_src.append(src_lab * result.slope + result.intercept)
    new_src = np.moveaxis(new_src, 0, -1)
    new_src = color.lab2rgb(new_src)
    return new_src

class LinearModel:
    def estimate(self, xs, ys, weights):
        total_weight = np.sum(weights)
        xs = xs * weights
        ys = ys * weights
        sx = np.sum(xs)
        sy = np.sum(ys)
        sxx = np.sum(np.square(xs))
        sxy = np.sum(xs * ys)
        slope_numer = total_weight * sxy - sx * sy
        slope_denom = total_weight * sxx - sx * sx
        self.slope = slope_numer / slope_denom
        self.intercept = (y - self.slope * x) / total_weight
    def residuals(self, xs, ys):
        ys_expected = self.slope * xs + self.intercept
        return np.sqrt(np.sum(np.square(ys_expected - ys), axis=1))
    def apply(self, xs):
        return self.slope * xs + self.intercept

@dataclass
class BinModel:
    data_min: float
    data_max: float
    bin_count: int
    def estimate(self, xs, ys, weights):
        xs_binned = self._bin(xs)
        self.bins = np.zeros((self.bin_count,))
        np.add.at(self.bins, xs_binned, ys * weights)
        bin_counts = np.zeros((bin_count,))
        np.add.at(bin_counts, xs_binned, weights)
        self.bins = np.where(bin_counts, self.bins / bin_counts, 0) # TODO: better choice here
    def residuals(self, xs, ys):
        ys_expected = self.apply(xs)
        return np.sqrt(np.sum(np.square(ys_expected - ys), axis=1))
    def apply(self, xs):
        return self.bins[self._bin(xs)]
    def _bin(self, xs):
        data_bins = (xs - self.data_min) / (self.data_max - self.data_min)
        data_bins = np.int_(data_bins * self.bin_count)
        data_bins = np.clip(data_bins, 0, self.bin_count - 1)
        return data_bins

def stream_from_file():
    src_stream = iio.imread('inputs/src-small.mp4')
    dst_stream = iio.imread('inputs/dst-small.mp4')
    while True:
        for src, dst in zip(src_stream, dst_stream):
            yield src, dst

def stream_two_camera():
    src = iio.imiter('<video0>', size='320x180')
    dst = iio.imiter('<video2>', size='320x180')
    for src, dst in zip(src, dst):
        yield src, dst

def stream_dummy():
    def recompress(frame):
        out = io.BytesIO()
        iio.imwrite(out, frame, extension='.jpeg')
        return iio.imread(out, extension='.jpeg')
    def split_thirds(frame):
        _, cols, _ = np.shape(frame)
        split1 = cols // 3
        split2 = 2 * cols // 3
        src = frame[::1, :split2:1, :]
        dst = frame[::1, split1::1, :]
        return src, dst
    rng = np.random.default_rng()
    for frame in iio.imiter('<video0>', size='320x180'):
        frame = util.img_as_float32(frame)
        src, dst = split_thirds(frame)
        dst = dst * 0.5
        noise_scale = 0.03
        src = src + rng.normal(scale=noise_scale, size=np.shape(src))
        dst = dst + rng.normal(scale=noise_scale, size=np.shape(dst))
        src = util.img_as_float32(recompress(util.img_as_ubyte(np.clip(src, 0, 1))))
        dst = util.img_as_float32(recompress(util.img_as_ubyte(np.clip(dst, 0, 1))))
        yield src, dst

def corresponding_coordinates(src, dst):
    src_gray = color.rgb2gray(src)
    dst_gray = color.rgb2gray(dst)

    extractor_src = feature.SIFT()
    extractor_src.detect_and_extract(src_gray)
    extractor_dst = feature.SIFT()
    extractor_dst.detect_and_extract(dst_gray)

    matches = feature.match_descriptors(extractor_src.descriptors, extractor_dst.descriptors, cross_check=True)
    model, inliers = measure.ransac((extractor_src.keypoints[matches[:, 0]], extractor_dst.keypoints[matches[:, 1]]),
                            transform.ProjectiveTransform, 
                            min_samples=8,
                            residual_threshold=1, 
                            max_trials=5000,
                            random_state=np.random.default_rng(seed=9))

    rows, cols, _ = np.shape(src)
    coords_src = np.reshape(np.mgrid[:rows, :cols], (2, -1))
    ones = np.ones((1, rows * cols))
    coords_dst = np.concatenate((coords_src, ones), axis=0)
    coords_dst = model.params @ coords_dst
    coords_dst = np.int_(coords_dst[:2, :] / coords_dst[2:, :])
    rows, cols, _ = np.shape(dst)
    valid = np.all((coords_dst >= 0, coords_dst < [[rows], [cols]]), axis=(0, 1))
    coords_src = coords_src[:,valid]
    coords_dst = coords_dst[:,valid]
    return coords_src, coords_dst

class Demo:
    def __init__(self, stream):
        stream = iter(stream)
        first_src, first_dst = next(stream)
        src_shape = np.shape(first_src)
        dst_shape = np.shape(first_dst)

        self.needs_calibration = True

        fig, axs = plt.subplots(1, 8)

        self.src_widget = axs[0].imshow(np.zeros(src_shape))
        self.src_widget.axes.set_position([0.0, 0.66, 0.3, 0.5])
        self.src_widget.axes.set_axis_off()

        self.dst_widget = axs[1].imshow(np.zeros(dst_shape))
        self.dst_widget.axes.set_position([0.0, 0.33, 0.3, 0.5])
        self.dst_widget.axes.set_axis_off()

        self.output_widget = axs[2].imshow(np.zeros(src_shape))
        self.output_widget.axes.set_position([0.0, 0.0, 0.3, 0.5])
        self.output_widget.axes.set_axis_off()

        self.src_points = axs[6].imshow(np.zeros(src_shape))
        self.src_points.axes.set_position([0.4, 0.6, 0.3, 0.5])
        self.src_points.axes.set_axis_off()

        self.dst_points = axs[7].imshow(np.zeros(src_shape))
        self.dst_points.axes.set_position([0.4, 0.3, 0.3, 0.5])
        self.dst_points.axes.set_axis_off()

        self.slider = widgets.Slider(axs[3], 'hello', 0, 1, valinit=0.3)
        self.slider.ax.set_position([0.6, 0.9, 0.5, 0.02]) 
        self.slider.on_changed(self.slider_update)

        self.textbox = widgets.TextBox(axs[4], 'world', initial='')
        self.textbox.ax.set_position([0.6, 0.6, 0.5, 0.02]) 
        self.textbox.on_submit(self.text_update)

        self.button = widgets.Button(axs[5], 'calibrate')
        self.button.ax.set_position([0.6, 0.3, 0.5, 0.02]) 
        self.button.on_clicked(self.calibrate_clicked)

        ani = animation.FuncAnimation(fig, self.animate, stream, cache_frame_data=False, blit=True, interval=50)
        plt.show()

    def text_update(self, new_text):
        pass

    def slider_update(self, new):
        pass
    
    def calibrate_clicked(self, arg):
        self.needs_calibration = True

    def animate(self, stream):
        src, dst = stream 
        
        if self.needs_calibration:
            self.needs_calibration = False
            self.coords_src, self.coords_dst = corresponding_coordinates(src, dst)
#             
#             self.lightness_model = IterativeReweight()
#             self.temperature_model = IterativeReweight()
#             self.tint_model = IterativeReweight()
# 
#         self.lightness_model.step()
#         self.temperature_model.step()
#         self.tint_model.step()

        output = estimate_model(src, dst, self.coords_src, self.coords_dst)
        
        self.src_widget.set_array(util.img_as_ubyte(src))
        self.dst_widget.set_array(util.img_as_ubyte(dst))
        self.output_widget.set_array(util.img_as_ubyte(output))
        return [self.src_widget, self.dst_widget, self.output_widget]

Demo(stream_dummy())


class IterativeReweight:
    def __init__(self, p, i, d, step, coords_src, coords_dst):
        initial_weights = rng.choice(len(data), min_weight, replace=False)
        self.weights = np.zeros((len(data), 1))
        self.weights[initial_weights] = 1.0
        self.error = 0
        self.integral = 0

    def estimate(src, dst, ):
        pass

    def step():
        model, residual = modeler(data, weights)
        factor = 1.345 * np.median(np.abs(residual - np.median(residual)))
        measurement = np.where(residual < factor, factor, factor / residual)

        step = 0.01

        setpoint = self.weights
        error = setpoint - measurement
        proportional = error
        self.integral = self.integral + error * self.step.val
        derivative = (error - self.error) / self.step.val
        self.error = error
        self.weights = self.p.val * proportional + self.i.val * self.integral + self.d.val * derivative

