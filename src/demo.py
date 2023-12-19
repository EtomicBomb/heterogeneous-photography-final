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
