from skimage import data
from skimage import transform
from skimage import io
from skimage.feature import match_descriptors, ORB, plot_matches, hog, SIFT
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.measure import ransac
import numpy as np
from skimage.transform import FundamentalMatrixTransform
from skimage import util
import sys

img1 = io.imread('right-small.png')
img2 = io.imread('left-small.png')
img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

def naive_search(source, texture):
    source = np.asarray(source, dtype=np.float32)
    texture = np.asarray(texture, dtype=np.float32)

    window_size = 20
    sample_count = 10
    source = np.sum(np.abs(np.gradient(source, axis=(0, 1))), axis=0)
    texture = np.sum(np.abs(np.gradient(texture, axis=(0, 1))), axis=0)
    print(source.shape, texture.shape)

    texture = util.view_as_windows(texture, (window_size, window_size, 3))
    texture = np.reshape(texture, (-1, window_size, window_size, 3))

    source_blocks = util.view_as_windows(source, (window_size, window_size, 3), (window_size, window_size, 1))
    source_blocks = np.reshape(source_blocks, (-1, window_size, window_size, 3))
    source_blocks = np.random.default_rng().choice(source_blocks, sample_count)
    source_blocks = np.reshape(source_blocks, (-1, 1, window_size, window_size, 3))

    for source_block, in source_blocks:
        result = np.square(source_block - texture)
        result = np.sum(result, axis=(1, 2, 3))
        result = np.argmin(result)
        print(result)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(source_block)a
        ax[1].imshow(texture[result])
        plt.show()

naive_search(img1, img2)
sys.exit(0)

# Find sparse feature correspondences between left and right image.

descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(img1_gray)
keypoints_left = descriptor_extractor.keypoints
descriptors_left = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2_gray)
keypoints_right = descriptor_extractor.keypoints
descriptors_right = descriptor_extractor.descriptors

matches = match_descriptors(descriptors_left, descriptors_right,
                            cross_check=True)

print(f'Number of matches: {matches.shape[0]}')

# Estimate the epipolar geometry between the left and right image.
random_seed = 9
rng = np.random.default_rng(random_seed)

model, inliers = ransac((keypoints_left[matches[:, 0]],
                         keypoints_right[matches[:, 1]]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000,
                        random_state=rng)

inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

print(f'Number of inliers: {inliers.sum()}')

# Compare estimated sparse disparities to the dense ground-truth disparities.

disp = inlier_keypoints_left[:, 1] - inlier_keypoints_right[:, 1]
disp_coords = np.round(inlier_keypoints_left).astype(np.int64)

# Visualize the results.

'''
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.gray()
plot_matches(ax, img1_gray, img2_gray, keypoints_left, keypoints_right,
             matches[inliers], only_matches=True)
ax.axis("off")
ax.set_title("Inlier correspondences")
plt.show()
'''
keypoints_left = np.asarray(keypoints_left, dtype=int)
keypoints_right = np.asarray(keypoints_right, dtype=int)


left_patches = []
right_patches = []

patch_size = 5

all_scales = []
all_offsets = []

for match_left, match_right in matches:
    r, c = keypoints_left[match_left]
    left_patch = img1[r-patch_size:r+patch_size,c-patch_size:c+patch_size,:]
    left_patches.append(left_patch)
    r, c = keypoints_right[match_right]
    right_patch = img2[r-patch_size:r+patch_size,c-patch_size:c+patch_size,:]
    right_patches.append(right_patch)

    # error = np.sum(np.square(np.subtract(np.gradient(np.mean(left_patch, axis=2)), np.gradient(np.mean(right_patch, axis=2)))))
    # print(error)

    l_grads, l_hog_image = hog(left_patch, orientations=8, pixels_per_cell=left_patch.shape[:2], cells_per_block=(1, 1), channel_axis=-1, visualize=True)
    r_grads, r_hog_image = hog(right_patch, orientations=8, pixels_per_cell=right_patch.shape[:2], cells_per_block=(1, 1), channel_axis=-1, visualize=True)
    error = np.mean(np.square(l_grads - r_grads))

    if error > 0.02: 
        continue

    # print(error)


    scales = []
    offsets = []

    left_ravel = np.reshape(left_patch, (-1, 3))
    right_ravel = np.reshape(right_patch, (-1, 3))
    for left_channel, right_channel in zip(np.split(left_ravel, 3, axis=1), np.split(right_ravel, 3, axis=1)):
        a = np.concatenate((left_channel, np.ones_like(left_channel)), axis=1)
        b = np.ravel(right_channel)
        (scale, offset), *_ = np.linalg.lstsq(a, b, rcond=None)
        scales.append(scale)
        offsets.append(offset)

    all_scales.append(scales)
    all_offsets.append(offsets)


#    fig, ax = plt.subplots(1, 5)
#    ax[0].imshow(left_patch)
#    ax[1].imshow(right_patch)
#    ax[2].imshow(np.clip((left_patch * scales + offsets)/255.0, 0, 1))
#    ax[3].imshow(r_hog_image)
#    ax[4].imshow(l_hog_image)
#    plt.show()

all_offsets = np.asarray(all_offsets)
all_scales = np.asarray(all_scales)


fig, ax = plt.subplots(1, 3)
ax[0].hist(all_scales[:,0], bins=20)
ax[1].hist(all_scales[:,1], bins=20)
ax[2].hist(all_scales[:,2], bins=20)
plt.show()

# calculate the mode of the scales and the mode of the offsets
def mode(a):
    bin_edges = np.linspace(np.min(a), np.max(a), num=20)
    print(bin_edges.shape)
    hist, _ = np.histogram(a, bin_edges)
    chosen_bin = np.argmax(hist)
    return (bin_edges[chosen_bin] + bin_edges[chosen_bin+1])/2
final_scales = []
final_offsets = []
for scale, off in zip(np.split(all_scales, 3, axis=1), np.split(all_offsets, 3, axis=1)):
    final_scales.append(mode(scale))
    final_offsets.append(mode(off))

adjusted = img1 * final_scales + final_offsets

print(scales, offsets)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[2].imshow(adjusted / 255.0)
plt.show()


'''

left_patches = np.asarray(np.reshape(left_patches,- (-1, 3)), dtype=np.float32)
right_patches = np.asarray(np.reshape(right_patches, (-1, 3)), dtype=np.float32)

scales = []
offsets = []

for left_channel, right_channel in zip(np.split(left_patches, 3, axis=1), np.split(right_patches, 3, axis=1)):
    a = np.concatenate((left_channel, np.ones_like(left_channel)), axis=1)
    b = np.ravel(right_channel)
    print(a, b)
    (scale, offset), *_ = np.linalg.lstsq(a, b)
    scales.append(scale)
    offsets.append(offset)

adjusted = img1 * scales + offsets

print(scales, offsets)

print(np.min(adjusted), np.max(adjusted), np.shape(adjusted))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[2].imshow(adjusted / 255.0)
plt.show()

'''


# tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
#                                   translation=(0, -200))
# img3 = transform.warp(img1_gray, tform)
#
# descriptor_extractor = SIFT()
#
# descriptor_extractor.detect_and_extract(img1_gray)
# keypoints1 = descriptor_extractor.keypoints
# descriptors1 = descriptor_extractor.descriptors
#
# descriptor_extractor.detect_and_extract(img2_gray)
# keypoints2 = descriptor_extractor.keypoints
# descriptors2 = descriptor_extractor.descriptors
#
# descriptor_extractor.detect_and_extract(img3)
# keypoints3 = descriptor_extractor.keypoints
# descriptors3 = descriptor_extractor.descriptors
#
# matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
# matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)
#
# fig, ax = plt.subplots(nrows=2, ncols=1)
#
# plt.gray()
#
# plot_matches(ax[0], img1_gray, img2_gray, keypoints1, keypoints2, matches12)
# ax[0].axis('off')
# ax[0].set_title("Original Image vs. Transformed Image")
#
# plot_matches(ax[1], img1_gray, img3, keypoints1, keypoints3, matches13)
# ax[1].axis('off')
# ax[1].set_title("Original Image vs. Transformed Image")
#
#
# plt.show()
#
