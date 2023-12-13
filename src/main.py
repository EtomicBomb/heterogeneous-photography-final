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

from bindings import *
    
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

# class LinearModel:
#     def estimate(self, xs, ys, weights):
#         total_weight = np.sum(weights)
#         xs = xs * weights
#         ys = ys * weights
#         sx = np.sum(xs)
#         sy = np.sum(ys)
#         sxx = np.sum(np.square(xs))
#         sxy = np.sum(xs * ys)
#         slope_numer = total_weight * sxy - sx * sy
#         slope_denom = total_weight * sxx - sx * sx
#         self.slope = slope_numer / slope_denom
#         self.intercept = (y - self.slope * x) / total_weight
#     def residuals(self, xs, ys):
#         ys_expected = self.slope * xs + self.intercept
#         return np.sqrt(np.sum(np.square(ys_expected - ys), axis=1))
#     def apply(self, xs):
#         return self.slope * xs + self.intercept

# @dataclass
# class BinModel:
#     data_min: float
#     data_max: float
#     bin_count: int
#     def estimate(self, xs, ys, weights):
#         xs_binned = self._bin(xs)
#         self.bins = np.zeros((self.bin_count,))
#         np.add.at(self.bins, xs_binned, ys * weights)
#         bin_counts = np.zeros((bin_count,))
#         np.add.at(bin_counts, xs_binned, weights)
#         self.bins = np.where(bin_counts, self.bins / bin_counts, 0) # TODO: better choice here
#     def residuals(self, xs, ys):
#         ys_expected = self.apply(xs)
#         return np.sqrt(np.sum(np.square(ys_expected - ys), axis=1))
#     def apply(self, xs):
#         return self.bins[self._bin(xs)]
#     def _bin(self, xs):
#         data_bins = (xs - self.data_min) / (self.data_max - self.data_min)
#         data_bins = np.int_(data_bins * self.bin_count)
#         data_bins = np.clip(data_bins, 0, self.bin_count - 1)
#         return data_bins

# def stream_from_file():
#     src_stream = iio.imread('inputs/src-small.mp4')
#     dst_stream = iio.imread('inputs/dst-small.mp4')
#     while True:
#         for src, dst in zip(src_stream, dst_stream):
#             src = util.img_as_float32(src)
#             dst = util.img_as_float32(dst)
#             yield src, dst

def stream_two_camera():
    src = iio.imiter('<video0>', size='320x180')
    dst = iio.imiter('<video2>', size='320x180')
    for src, dst in zip(src, dst):
        yield src, dst

def stream_two_still():
    src = iio.imread('inputs/Rocks2/view1.png')
    dst = iio.imread('inputs/Rocks2/view5.png')
    src = util.img_as_float32(src)
    dst = util.img_as_float32(dst)
#     src = src[:,:,:3]
#     dst = dst[:,:,:3]
    while True:
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
#     while True: 
#         frame = iio.imread('inputs/20231111_0002.jpg')
    for frame in iio.imiter('<video0>', size='320x180'):
        frame = util.img_as_float32(frame)
        src, dst = split_thirds(frame)
        dst = dst * 0.5
        noise_scale = 0.0
        src = src + rng.normal(scale=noise_scale, size=np.shape(src))
        dst = dst + rng.normal(scale=noise_scale, size=np.shape(dst))
        src = util.img_as_float32(recompress(util.img_as_ubyte(np.clip(src, 0, 1))))
        dst = util.img_as_float32(recompress(util.img_as_ubyte(np.clip(dst, 0, 1))))
        yield src, dst


def corresponding_coordinates(src, dst, max_ratio):
    src_gray = color.rgb2gray(src)
    dst_gray = color.rgb2gray(dst)

    extractor_src = feature.SIFT()
    extractor_src.detect_and_extract(src_gray)
    extractor_dst = feature.SIFT()
    extractor_dst.detect_and_extract(dst_gray)

    matches = feature.match_descriptors(extractor_src.descriptors, extractor_dst.descriptors, cross_check=True, max_ratio=max_ratio)

    fig, axs = plt.subplots()
    feature.plot_matches(axs, src, dst, extractor_src.keypoints, extractor_dst.keypoints, matches)

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

#         fig = plt.figure(figsize=(4, 6), layout="constrained")
#         gs0 = fig.add_gridspec(6, 2)
#         ax1 = fig.add_subplot(gs0[:3, 0])
#         ax2 = fig.add_subplot(gs0[3:, 0])

        stream = iter(stream)
        first_src, first_dst = next(stream)
        src_shape = np.shape(first_src)
        dst_shape = np.shape(first_dst)

        self.needs_calibration = True

        fig = plt.figure(num='streams', layout='constrained')
        gs = GridSpec(15, 3, figure=fig)

        ax = fig.add_subplot(gs[0:3, 0])
        self.src_widget = ax.imshow(np.zeros(src_shape))
        self.src_widget.axes.set_axis_off()

        ax = fig.add_subplot(gs[0:3, 1])
        self.dst_widget = ax.imshow(np.zeros(dst_shape))
        self.dst_widget.axes.set_axis_off()

        ax = fig.add_subplot(gs[0:3, 2])
        self.output_widget = ax.imshow(np.zeros(src_shape))
        self.output_widget.axes.set_axis_off()

        ax = fig.add_subplot(gs[3:6, 0])
        self.src_points = ax.imshow(np.zeros(src_shape))
        self.src_points.axes.set_axis_off()

        ax = fig.add_subplot(gs[3:6, 1])
        self.dst_points = ax.imshow(np.zeros(src_shape))
        self.dst_points.axes.set_axis_off()

        ax = fig.add_subplot(gs[-1, 2])
        self.patch_size = widgets.Slider(ax, 'patch size', 1, 50, valinit=25)
        self.patch_size.on_changed(self.slider_update)

        ax = fig.add_subplot(gs[-2, 2])
        self.occlusion_cost = widgets.Slider(ax, 'occlusion cost', 0, 1, valinit=0.02)
        self.occlusion_cost.on_changed(self.slider_update)

        ax = fig.add_subplot(gs[-1, 1])
        self.textbox = widgets.TextBox(ax, 'world', initial='')
        self.textbox.on_submit(self.text_update)

        ax = fig.add_subplot(gs[-1, 0])
        self.button = widgets.Button(ax, 'calibrate')
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
#             self.coords_src, self.coords_dst = corresponding_coordinates(src, dst, 1.0)
#             self.h1, self.h2 = rectify(src, dst, 0.4)

#         src_rectify = cv2.warpPerspective(src, self.h1, np.shape(src)[:2])
#         dst_rectify = cv2.warpPerspective(src, self.h2, np.shape(src)[:2])
#         self.src_points.set_array(util.img_as_ubyte(src_rectify))
#         self.dst_points.set_array(util.img_as_ubyte(dst_rectify))

        patch_size = np.int_(self.patch_size.val)
        occlusion_cost = self.occlusion_cost.val
        num_threads = 8
        src_gray = color.rgb2gray(src)
        dst_gray = color.rgb2gray(dst)
        correspondence, valid, timings = scanline_stereo_cpu(src_gray, dst_gray, patch_size, occlusion_cost, num_threads)
        valid = np.expand_dims(valid, -1)
        correspondence = np.arange(src_gray.shape[1]) - correspondence
        correspondence = np.expand_dims(correspondence, -1) * [1.0, 1.0, 1.0]
        correspondence = (correspondence - np.min(correspondence)) / np.ptp(correspondence)
        self.src_points.set_array(util.img_as_ubyte(np.ones_like(src) * valid))
        self.dst_points.set_array(util.img_as_ubyte(correspondence * [1.0, 1.0, 1.0]))

#         output = estimate_model(src, dst, self.coords_src, self.coords_dst)
#         output = src

        self.src_widget.set_array(util.img_as_ubyte(src))
        self.dst_widget.set_array(util.img_as_ubyte(dst))
        self.output_widget.set_array(util.img_as_ubyte(src))

        return [self.src_widget, self.dst_widget, self.output_widget, self.src_points, self.dst_points]

def rectify(src, dst, max_ratio):
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(src, None)
#     kp2, des2 = sift.detectAndCompute(dst, None)
#     matches = feature.match_descriptors(des1, des2, cross_check=True, max_ratio=max_ratio)
#     print(len(matches))
#     kp1 = np.array([p.pt for p in kp1])
#     kp2 = np.array([p.pt for p in kp2])
#     print(kp1)
#     points_left = kp1[matches[:, 0]]
#     points_right = kp2[matches[:, 1]]

    extractor_src = feature.SIFT()
    extractor_src.detect_and_extract(src_gray)
    extractor_dst = feature.SIFT()
    extractor_dst.detect_and_extract(dst_gray)
    matches = feature.match_descriptors(extractor_src.descriptors, extractor_dst.descriptors, cross_check=True, max_ratio=max_ratio)
    points_left = np.fliplr(extractor_src.keypoints[matches[:, 0]])
    points_right = np.fliplr(extractor_dst.keypoints[matches[:, 1]])

    F, mask = cv2.findFundamentalMat(points_left, points_right)
#     F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC, ransacReprojThreshold=1, confidence=0.99)

#     F, inliers = measure.ransac((points_left, points_right),
#                             transform.FundamentalMatrixTransform, 
#                             min_samples=8,
#                             residual_threshold=1, 
#                             max_trials=5000,
#                             random_state=np.random.default_rng(seed=9)) # row, col -> row, col
#     F = F.params

    success, h1, h2 = cv2.stereoRectifyUncalibrated(points_left, points_right, F, src_gray.shape)
    assert success

    src_rectify = cv2.warpPerspective(src, h1, np.flip(np.shape(src)[:2]))
    dst_rectify = cv2.warpPerspective(dst, h2, np.flip(np.shape(dst)[:2]))
    h1 = transform.ProjectiveTransform(matrix=h1)
    h2 = transform.ProjectiveTransform(matrix=h2)
    src_rectify0 = transform.warp(src, h1.inverse, output_shape=None, order=None, cval=0.0)
    dst_rectify0 = transform.warp(dst, h2.inverse, output_shape=None, order=None, cval=0.0)

    fig, axs = plt.subplots(2, 3)
    axs = np.ravel(axs)
    feature.plot_matches(axs[0], src, dst, extractor_src.keypoints, extractor_dst.keypoints, matches)
    axs[1].imshow(src_rectify)
    axs[2].imshow(dst_rectify)
    axs[3].imshow(src_rectify0)
    axs[4].imshow(dst_rectify0)
    plt.show()

    return src_rectify0, dst_rectify0

def rectify_demo(left, right):
    left_gray = color.rgb2gray(left)
    right_gray = color.rgb2gray(right)
    left_gray = util.img_as_ubyte(left_gray)
    right_gray = util.img_as_ubyte(right_gray)

    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(left_gray, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L1)
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.18 * n.distance]

    source_points = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(source_points, destination_points, cv2.FM_RANSAC, 5.0)

    image_size = np.flip(left_gray.shape[:2])

    success, H1, H2 = cv2.stereoRectifyUncalibrated(source_points, destination_points, F, image_size)
    assert success

    rectified_left = cv2.warpPerspective(left, H1, image_size)
    rectified_right = cv2.warpPerspective(right, H2, image_size)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(rectified_left)
    axs[1].imshow(rectified_right)
    plt.show()

    return rectified_left, rectified_right

# left = iio.imread('inputs/Rocks2/view1.png')
# right = iio.imread('inputs/Rocks2/view5.png')
# print(rectify(left, right, 0.6))
# print(rectify_demo(left, right))
# src_small = color.rgb2gray(left)
# dst_small = color.rgb2gray(right)
# patch_size = 10
# occlusion_cost = 0.01
# num_threads = 8
# correspondence, valid, timings = scanline_stereo_cpu(src_small, dst_small, patch_size, occlusion_cost, num_threads)
# fig, axs = plt.subplots(2, 2)
# axs = np.ravel(axs)
# axs[0].imshow(np.arange(src_small.shape[1]) - correspondence, cmap='Greys', vmin=10, vmax=60)
# axs[1].imshow(valid)
# axs[2].imshow(left)
# axs[3].imshow(right)
# plt.show()

Demo(stream_two_still())


#             
#             self.lightness_model = IterativeReweight()
#             self.temperature_model = IterativeReweight()
#             self.tint_model = IterativeReweight()
# 
#         self.lightness_model.step()
#         self.temperature_model.step()
#         self.tint_model.step()

# class IterativeReweight:
#     def __init__(self, p, i, d, step, coords_src, coords_dst):
#         initial_weights = rng.choice(len(data), min_weight, replace=False)
#         self.weights = np.zeros((len(data), 1))
#         self.weights[initial_weights] = 1.0
#         self.error = 0
#         self.integral = 0

#     def estimate(src, dst, ):
#         pass

#     def step():
#         model, residual = modeler(data, weights)
#         factor = 1.345 * np.median(np.abs(residual - np.median(residual)))
#         measurement = np.where(residual < factor, factor, factor / residual)

#         step = 0.01

#         setpoint = self.weights
#         error = setpoint - measurement
#         proportional = error
#         self.integral = self.integral + error * self.step.val
#         derivative = (error - self.error) / self.step.val
#         self.error = error
#         self.weights = self.p.val * proportional + self.i.val * self.integral + self.d.val * derivative

