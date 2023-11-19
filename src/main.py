import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from skimage import data, transform, util, io, color, measure, feature, registration
import cv2
import imageio.v3 as iio

import bindings

def corresponding_coordinates(left, right):
    left_gray = color.rgb2gray(left)
    right_gray = color.rgb2gray(right)

    extractor_left = feature.SIFT()
    extractor_left.detect_and_extract(left_gray)
    extractor_right = feature.SIFT()
    extractor_right.detect_and_extract(right_gray)

    matches = feature.match_descriptors(extractor_left.descriptors, extractor_right.descriptors, cross_check=True)
    model, inliers = measure.ransac((extractor_left.keypoints[matches[:, 0]], extractor_right.keypoints[matches[:, 1]]),
                            transform.ProjectiveTransform, 
                            min_samples=8,
                            residual_threshold=1, 
                            max_trials=5000,
                            random_state=np.random.default_rng(seed=9))

    rows, cols, _ = np.shape(left)
    coords_left = np.reshape(np.mgrid[:rows, :cols], (2, -1))
    ones = np.ones((1, rows * cols))
    coords_right = np.concatenate((coords_left, ones), axis=0)
    coords_right = model.params @ coords_right
    coords_right = np.int_(coords_right[:2, :] / coords_right[2:, :])
    rows, cols, _ = np.shape(right)
    to_delete = np.logical_or(coords_right < 0, coords_right >= [[rows], [cols]])
    to_delete = np.any(to_delete, axis=0)
    coords_left = np.delete(coords_left, to_delete, axis=1)
    coords_right = np.delete(coords_right, to_delete, axis=1)

    return coords_left, coords_right

def corresponding_coordinates_flow(left, right):
    left = color.rgb2gray(left)
    right = color.rgb2gray(right)

    rows, cols = np.shape(left)
    coords_left = np.mgrid[:rows, :cols]
    coords_left = np.reshape(coords_left, (2, -1))

    coords_right = registration.optical_flow_ilk(left, right)
    coords_right = np.reshape(coords_right, (2, -1))

    coords_right = coords_left + coords_right
    coords_right = np.int_(np.round(coords_right))
    to_delete = np.logical_or(coords_right < 0, coords_right >= [[rows], [cols]])
    to_delete = np.any(to_delete, axis=0)
    coords_left = np.delete(coords_left, to_delete, axis=1)
    coords_right = np.delete(coords_right, to_delete, axis=1)

    return coords_left, coords_right

def estimate_model(left, right, coords_left, coords_right):
    colors_left = left[coords_left[0], coords_left[1], :]
    colors_left = color.rgb2lab(colors_left)
    colors_right = right[coords_right[0], coords_right[1], :]
    colors_right = color.rgb2lab(colors_right)
    to_delete = np.any((colors_left[:,0] < 10, colors_left[:,0] > 90, colors_right[:,0] < 10, colors_right[:,0] > 90), axis=0)
    print('filtering', np.sum(to_delete), 'elements to do over- or under-exposure')
    colors_left = np.delete(colors_left, to_delete, axis=0)
    colors_right = np.delete(colors_right, to_delete, axis=0)

    new_left = []
    left_lab = color.rgb2lab(left)
    for left_lab, channel_left, channel_right in zip(np.moveaxis(left_lab, -1, 0), np.transpose(colors_left), np.transpose(colors_right)):
        print(channel_left.shape, channel_right.shape)
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].scatter(channel_left, channel_right)
        result = scipy.stats.linregress(channel_left, channel_right)
        new_left.append(left_lab * result.slope + result.intercept)
        axs[1].scatter(channel_left, channel_left * result.slope + result.intercept)
        plt.show()
#         model, inliers = measure.ransac(np.column_stack((channel_left, channel_right)),
#                         measure.LineModelND, 
#                         min_samples=8,
#                         residual_threshold=0.003, 
#                         max_trials=500,
#                         random_state=np.random.default_rng(seed=9))
#         print(len(inliers))
#         axs[1].scatter(channel_left, model.predict_y(channel_left))
#         plt.show()
#         new_left.append(np.reshape(model.predict_y(np.ravel(left_lab)), np.shape(left_lab)))
    new_left = np.moveaxis(new_left, 0, -1)
    new_left = color.lab2rgb(new_left)
    return new_left

def modeler_linear_regression(data, weights):
    weighted = data * weights
    total_weight = np.sum(weights)

    x, y = np.sum(weighted, axis=0)
    xx, yy = np.sum(np.square(weighted), axis=0)
    xy = np.sum(weighted[:,0] * weighted[:, 1])

    slope_numer = total_weight * xy - x * y
    slope_denom = total_weight * xx - x * x
    slope = slope_numer / slope_denom
    intercept = y - slope * x
    intercept = intercept / total_weight

    residual = np.square(data[:,0] * slope + intercept - data[:,1])

    return (slope, intercept), residual

def modeler_bins(data, weights, bin_count, data_min, data_max):
    data_bins = (data[:,0] - data_min) / (data_max - data_min)
    data_bins = np.int_(data_bins * bin_count)
    data_bins = np.clip(data_bins, 0, bin_count - 1)
    
    bins = np.zeros((bin_count,))
    np.add.at(bins, data_bins, data[:,1] * weights)
    bin_counts = np.zeros((bin_count,))
    np.add.at(bin_counts, data_bins, weights)

    bins = np.where(bin_counts, bins / bin_counts, 0) # TODO: better choice here

    residual = np.square(bins[data_bins] - data[:,1])

    return bins, residual

def iteratively_reweighted(data, modeler, min_weight, iteration_count, rng):
    initial_weights = rng.choice(len(data), min_weight, replace=False)
    weights = np.zeros((len(data), 1))
    weights[initial_weights] = 1.0

    while iteration_count:
        model, residual = modeler(data, weights)
        factor = 1.345 * np.median(np.abs(residual - np.median(residual)))
        new_weights = np.where(residual < factor, factor, factor / residual)
        weights = new_weights

    return model

def webcam_demo():
    # webcam_stream = iio.imiter('inputs/video.mp4')
    webcam_stream = (a[::1, ::1, :] for a in iio.imiter('<video0>', size='320x180'))
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    slider = widgets.Slider(ax2, 'hello', 0, 1, valinit=0.3)
    slider.ax.set_position([0.25, 0.01, 0.5, 0.02]) 
    slider.on_changed(lambda v: print(v))
    textbox = widgets.TextBox(ax3, 'world', initial='')
    textbox.ax.set_position([0.25, 0.11, 0.5, 0.02]) 
    textbox.on_submit(lambda v: print(v))

    ax_image = ax.imshow(next(webcam_stream))
    def animate(image):
        ax_image.set_array(np.uint8(image * slider.val))
        return [ax_image]
    ani = animation.FuncAnimation(fig, animate, webcam_stream, cache_frame_data=False, blit=True, interval=0)
    plt.show()

left = io.imread('inputs/a.jpg')
right = io.imread('inputs/b.jpg')

left = util.img_as_float32(left)
right = util.img_as_float32(right)

webcam_demo()

coords_left, coords_right = corresponding_coordinates(left, right)
# output = estimate_model(left, right, coords_left, coords_right)
# print(np.min(output), np.max(output))
rl, cl = coords_left
rr, cr = coords_right
left[rl, cl, :] *= 0.5
right[rr, cr, :] *= 0.5
fig, ax = plt.subplots(1, 2)
ax[0].imshow(left)
ax[1].imshow(right)
plt.show()
