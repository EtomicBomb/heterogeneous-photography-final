import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib as mpl

stage_labels = ['pixel similarity', 'sum rows', 'sum cols src', 'sum cols dst', 'patch similarity', 'costs', 'traceback']
image_sizes = [139, 278, 416, 555]

# gpu
gpu_data = [
    [0.17816, 0.827392, 0.178176, 0.356816, 0.31984, 0.373488, 0.134992],
    [0.562208, 3.037296, 1.432624, 1.4772799, 1.255536, 7.599696, 0.238224],
    [0.94542396, 5.2615843, 2.702352, 2.622032, 2.193664, 15.150145, 0.344064],
    [2.200288, 12.603216, 6.2593603, 9.775696, 5.07616, 37.055138, 0.559968],
]

# cpu
cpu_data = [
    [1.0, 1.0, 1.0, 1.0, 4.0, 3.0, 0.0],
    [7.5, 12.5, 6.5, 6.5, 21.0, 19.5, 0.0],
    [13.0, 23.0, 12.0, 11.0, 38.0, 35.0, 0.0],
    [29.0, 47.0, 29.0, 25.0, 85.5, 123.0, 0.5],
]



# Plotting GPU chart with horizontal bars
fig, axs = plt.subplots(2, 1, sharex=True)
ax = axs[0]
bar_height = 0.2
for i, size in enumerate(image_sizes):
    ax.barh(np.arange(len(stage_labels)) + i * bar_height, cpu_data[i], bar_height, label=f'image size: {size} rows')

ax.set_yticks(np.arange(len(stage_labels)) + (len(image_sizes) - 1) * bar_height / 2)
ax.set_yticklabels(stage_labels)
ax.set_xlabel('runtime (milliseconds)')
ax.set_title('time to solution, cpu implementation')
ax.legend()

ax = axs[1]

bar_height = 0.2
for i, size in enumerate(image_sizes):
    ax.barh(np.arange(len(stage_labels)) + i * bar_height, gpu_data[i], bar_height, label=f'image size: {size} rows')

ax.set_yticks(np.arange(len(stage_labels)) + (len(image_sizes) - 1) * bar_height / 2)
ax.set_yticklabels(stage_labels)
ax.set_xlabel('runtime (milliseconds)')
ax.set_title('time to solution, gpu implementation')
ax.legend()

plt.show()
