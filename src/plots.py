import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib as mpl

data = defaultdict(lambda: defaultdict(list))

stage_labels = ['pixel_similarity', 'sum_rows', 'sum_cols1', 'sum_cols2', 'patch_similarity', 'costs', 'traceback', 'patch_similarity2']
for line in open('reports/timings.txt', 'r').readlines():
    patch_size, rows, cols_src, cols_dst, *stage_times = map(float, line.strip().split(' '))
    for stage_time, stage_label in zip(stage_times, stage_labels):
        data[stage_label][patch_size].append((rows * cols_src * cols_dst, stage_time))

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

fig, axs = plt.subplots(1, 3, dpi=80, sharex=True, sharey=True)
axs = iter(np.ravel(axs))

first = True
for stage_label, patch_size_to_pixel_time in data.items():
    
    print(stage_label)
    if stage_label not in ('costs', 'traceback', 'patch_similarity2'):
        continue
    ax = next(axs)

    x = np.arange(3)  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    
    for patch_size, pixel_time in patch_size_to_pixel_time.items():
        offset = width * multiplier
        pixel_time = np.array(pixel_time)[[0, 3, 4]]
        pixel, time = zip(*pixel_time)

        print(pixel, time)
        bar_container = ax.barh(x + offset, list(time), width, label=f'patch_size={patch_size}')
        ax.bar_label(bar_container, fmt='{:,.2f}')
        #ax.plot(pixel, time, label=f'patch_size={patch_size}')
        multiplier += 1

    #ax.set_yticks(x+width,  ['small', 'medium', 'large'])
    ax.set_yticks(x+width,  ['65px', '208px', '370px'])
    ax.set(title=stage_label)
    ax.semilogx()
    ax.set_xlim((0.01, 190))

    

    if first:
        first = False
        ax.set(xlabel='log median execution time (ms)')
        ax.legend(loc='lower right', ncols=2)
plt.show()


'''
first = True

    data = sorted(data, key=lambda x: x[0] * x[1])

    bar_data = []
    for rows, cols, time in data:
        bar_data.append((f'{int(rows)}x{int(cols)}', time))
    titles, max_times = zip(*bar_data)

    bar_container = ax.barh(list(titles), list(max_times))
    ax.semilogx()
    ax.set_xlim((1000, 6000))
    ax.bar_label(bar_container, fmt='{:,.0f}')
    ax.set(ylabel=f'{int(stream)} streams', title=stage_label)

    if first:
        ax.set(xlabel='log execution time (microseconds)')
        first = False
'''

# subplot_width = math.ceil(math.sqrt(len(combinations)))
# fig, axs = plt.subplots(subplot_width, subplot_width, sharex=True, sharey=True)
# axs = np.ravel(axs)
#     ax.set_title(title)
#     ax.loglog(dims, times)
