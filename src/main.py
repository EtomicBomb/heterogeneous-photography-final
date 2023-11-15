import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, util, io, color, measure, feature
from skimage.transform import FundamentalMatrixTransform, ProjectiveTransform, EuclideanTransform
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
                            ProjectiveTransform, 
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

#     fig, ax = plt.subplots(1, 3)
#     ax[0].imshow(left)
#     ax[1].imshow(right)
#     right_copy = np.array(right)
#     right_copy[coords_right[0], coords_right[1], :] = [255, 0, 0]
#     ax[2].imshow(right_copy)
#     plt.show()

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
        model, inliers = measure.ransac(np.column_stack((channel_left, channel_right)),
                        measure.LineModelND, 
                        min_samples=8,
                        residual_threshold=0.003, 
                        max_trials=500,
                        random_state=np.random.default_rng(seed=9))
        print(len(inliers))
        new_left.append(np.reshape(model.predict_y(np.ravel(left_lab)), np.shape(left_lab)))
    new_left = np.moveaxis(new_left, 0, -1)
    new_left = color.lab2rgb(new_left)
    return new_left

left = io.imread('inputs/a.jpg')
left = util.img_as_float32(left)
right = io.imread('inputs/b.jpg')
right = util.img_as_float32(right)

coords_left, coords_right = corresponding_coordinates(left, right)
output = estimate_model(left, right, coords_left, coords_right)
print(np.min(output), np.max(output))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(left)
ax[1].imshow(right)
ax[2].imshow(output)
plt.show()
