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

from bindings import *

def rectify(src, dst, max_ratio):
    src_gray = color.rgb2gray(src)
    dst_gray = color.rgb2gray(dst)

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

    F, inliers = measure.ransac((points_left, points_right),
                            transform.FundamentalMatrixTransform, 
                            min_samples=8,
                            residual_threshold=1, 
                            max_trials=5000,
                            random_state=np.random.default_rng(seed=9))
    F = F.params
    success, h1, h2 = cv2.stereoRectifyUncalibrated(points_left, points_right, F, src_gray.shape)
#     assert success
    h1 = transform.ProjectiveTransform(matrix=h1).inverse
    h2 = transform.ProjectiveTransform(matrix=h2).inverse
    return h1, h2

def corrupt(rows, cols, darken, tint, temperature, noise_scale, noise_count):
    def ret(image):
        l, a, b = np.moveaxis(color.rgb2lab(image), 2, 0)
        l = np.where(abs(l - 50) < 49, l * darken, l)
        a = np.clip(a + tint, -128, 127)
        b = np.clip(b + temperature, -128, 127)
        image = np.stack((l, a, b), axis=2)
        image = color.lab2rgb(image)

        rng = np.random.default_rng()
        image = image + rng.normal(scale=noise_scale, size=(rows, cols, 3))
        image = np.clip(image, 0, 1)

        image = util.img_as_ubyte(image)

        buffer = io.BytesIO()
        iio.imwrite(buffer, image, extension='.jpeg')
        image = iio.imread(buffer, extension='.jpeg')

        return image
    return ret
    
def estimate_model(src, dst, coords_src, coords_dst):
    colors_src = src[coords_src[0], coords_src[1], :]
    colors_src = color.rgb2lab(colors_src)
    colors_dst = dst[coords_dst[0], coords_dst[1], :]
    colors_dst = color.rgb2lab(colors_dst)
    valid = np.all((colors_src[:,0] > 10, colors_src[:,0] < 90, colors_dst[:,0] > 10, colors_dst[:,0] < 90), axis=0)
    colors_src = colors_src[valid, :]
    colors_dst = colors_dst[valid, :]
    if len(colors_src) < 10:
        print('not enough matches found')
        return dst
    new_dst = []
    lab_dst = np.moveaxis(color.rgb2lab(dst), -1, 0)
    for lab_dst, channel_src, channel_dst in zip(lab_dst, np.transpose(colors_src), np.transpose(colors_dst)):
        result = scipy.stats.linregress(channel_src, channel_dst)
        new_dst.append(lab_dst / result.slope - result.intercept / result.slope)
    return color.lab2rgb(np.moveaxis(new_dst, 0, -1))

# def corresponding_coordinates(src, dst, max_ratio):
#     src_gray = color.rgb2gray(src)
#     dst_gray = color.rgb2gray(dst)
# 
#     extractor_src = feature.SIFT()
#     extractor_src.detect_and_extract(src_gray)
#     extractor_dst = feature.SIFT()
#     extractor_dst.detect_and_extract(dst_gray)
# 
#     matches = feature.match_descriptors(extractor_src.descriptors, extractor_dst.descriptors, cross_check=True, max_ratio=max_ratio)
# 
#     fig, axs = plt.subplots()
#     feature.plot_matches(axs, src, dst, extractor_src.keypoints, extractor_dst.keypoints, matches)
# 
#     model, inliers = measure.ransac((extractor_src.keypoints[matches[:, 0]], extractor_dst.keypoints[matches[:, 1]]),
#                             transform.ProjectiveTransform, 
#                             min_samples=8,
#                             residual_threshold=1, 
#                             max_trials=5000,
#                             random_state=np.random.default_rng(seed=9))
# 
#     rows, cols, _ = np.shape(src)
#     coords_src = np.reshape(np.mgrid[:rows, :cols], (2, -1))
#     ones = np.ones((1, rows * cols))
#     coords_dst = np.concatenate((coords_src, ones), axis=0)
#     coords_dst = model.params @ coords_dst
#     coords_dst = np.int_(coords_dst[:2, :] / coords_dst[2:, :])
#     rows, cols, _ = np.shape(dst)
#     valid = np.all((coords_dst >= 0, coords_dst < [[rows], [cols]]), axis=(0, 1))
#     coords_src = coords_src[:,valid]
#     coords_dst = coords_dst[:,valid]
#     return coords_src, coords_dst


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

# # 2 matrices of floats (images)
# def foo(src, dst):
#     src = iio.imread('inputs/a.jpg')
#     dst = iio.imread('inputs/b.jpg')
# 
#     src = util.img_as_float32(src)
#     dst = util.img_as_float32(dst)
# 
#     src_gray = color.rgb2gray(src)
#     dst_gray = color.rgb2gray(dst)
# 
#     extractor_src = feature.SIFT()
#     extractor_src.detect_and_extract(src_gray)
#     extractor_dst = feature.SIFT()
#     extractor_dst.detect_and_extract(dst_gray)
# 
#     matches = feature.match_descriptors(extractor_src.descriptors, extractor_dst.descriptors, cross_check=True)
#     essential, inliers = measure.ransac((extractor_src.keypoints[matches[:, 0]], extractor_dst.keypoints[matches[:, 1]]),
#                             transform.EssentialMatrixTransform, 
#                             min_samples=8,
#                             residual_threshold=1, 
#                             max_trials=5000,
#                             random_state=np.random.default_rng(seed=9))
# 
#     src_norm = src_gray - np.mean(src_gray)
#     dst_norm = dst_gray - np.mean(dst_gray)
# 
#     rows, cols, _ = src.shape
#     row, col = 15, 30
#     x = [row, col, 1]
#     x = np.reshape(x, (3, 1))
#     ex = essential.params @ x
# 
#     src_patch = src_norm[row-10:row+10, col-10:col+10]
# 
#     rows, cols, _ = dst.shape
#     indices = np.mgrid[:rows, :cols]
#     indices = np.concatenate((indices, np.ones((1, rows, cols))), axis=0)
#     weights = np.expand_dims(ex, 2) * indices
#     weights = np.sum(weights, axis=0)
#     sigma = 0.04
#     weights = np.exp(-np.square(weights) / (2 * np.square(sigma)))
#     
#     correlation = scipy.signal.correlate2d(dst_norm, src_patch, mode='same')
# 
#     print(weights.shape, correlation.shape, dst.shape)
# 
#     # x'Ex = 0 => x' [l1, l2, l3] = 0 => l1 x1 + l2 y1 + l3 = 0
#     xs = np.arange(cols)
#     l1, l2, l3 = ex
#     ys = (-l3 - l1*xs)/l2
#     valid = np.logical_and(ys >= 0, ys < rows)
#     xs = np.int_(xs[valid])
#     ys = np.int_(ys[valid])
#     print(xs)
#     print(ys)
# 
#     dst[ys, xs, :] = [255, 0, 0]
#     fig, ax = plt.subplots(1, 5)
#     ax[0].imshow(src)
#     ax[1].imshow(dst)
#     ax[2].imshow(weights)
#     ax[3].imshow(correlation)
#     ax[4].imshow(weights * correlation)
# 
#     plt.show()

# def stream_from_file():
#     src_stream = iio.imread('inputs/src-small.mp4')
#     dst_stream = iio.imread('inputs/dst-small.mp4')
#     while True:
#         for src, dst in zip(src_stream, dst_stream):
#             src = util.img_as_float32(src)
#             dst = util.img_as_float32(dst)
#             yield src, dst

# def stream_two_camera():
#     src = iio.imiter('<video0>', size='320x180')
#     dst = iio.imiter('<video2>', size='320x180')
#     for src, dst in zip(src, dst):
#         yield src, dst

# def stream_two_still():
#     src = iio.imread('inputs/half/Lampshade1/view1.png')
#     dst = iio.imread('inputs/half/Lampshade1/view5.png')
#     src = util.img_as_float32(src)
#     dst = util.img_as_float32(dst)
# #     src = src[:,:,:3]
# #     dst = dst[:,:,:3]
#     while True:
#         yield src, dst

# def stream_dummy():
#     def recompress(frame):
#         out = io.BytesIO()
#         iio.imwrite(out, frame, extension='.jpeg')
#         return iio.imread(out, extension='.jpeg')
#     def split_thirds(frame):
#         _, cols, _ = np.shape(frame)
#         split1 = cols // 3
#         split2 = 2 * cols // 3
#         src = frame[::1, :split2:1, :]
#         dst = frame[::1, split1::1, :]
#         return src, dst
#     rng = np.random.default_rng()
# #     while True: 
# #         frame = iio.imread('inputs/20231111_0002.jpg')
#     for frame in iio.imiter('<video0>', size='320x180'):
#         frame = util.img_as_float32(frame)
#         src, dst = split_thirds(frame)
#         dst = dst * 0.5
#         noise_scale = 0.0
#         src = src + rng.normal(scale=noise_scale, size=np.shape(src))
#         dst = dst + rng.normal(scale=noise_scale, size=np.shape(dst))
#         src = util.img_as_float32(recompress(util.img_as_ubyte(np.clip(src, 0, 1))))
#         dst = util.img_as_float32(recompress(util.img_as_ubyte(np.clip(dst, 0, 1))))
#         yield src, dst

# def rectify_demo(left, right):
#     left_gray = color.rgb2gray(left)
#     right_gray = color.rgb2gray(right)
#     left_gray = util.img_as_ubyte(left_gray)
#     right_gray = util.img_as_ubyte(right_gray)
# 
#     sift = cv2.SIFT_create()
#     keypoints_left, descriptors_left = sift.detectAndCompute(left_gray, None)
#     keypoints_right, descriptors_right = sift.detectAndCompute(right_gray, None)
# 
#     bf = cv2.BFMatcher(cv2.NORM_L1)
#     matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)
# 
#     good_matches = [m for m, n in matches if m.distance < 0.18 * n.distance]
# 
#     source_points = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     destination_points = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# 
#     F, mask = cv2.findFundamentalMat(source_points, destination_points, cv2.FM_RANSAC, 5.0)
# 
#     image_size = np.flip(left_gray.shape[:2])
# 
#     success, H1, H2 = cv2.stereoRectifyUncalibrated(source_points, destination_points, F, image_size)
#     assert success
# 
#     rectified_left = cv2.warpPerspective(left, H1, image_size)
#     rectified_right = cv2.warpPerspective(right, H2, image_size)
# 
#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(rectified_left)
#     axs[1].imshow(rectified_right)
#     plt.show()
# 
#     return rectified_left, rectified_right
