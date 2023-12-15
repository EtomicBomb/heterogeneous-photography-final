import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib as mpl

# 1 370 417 417 32.0 187.0 88.0 1.0
# 3 370 417 417 32.0 476.5 88.0 1.0
# 9 370 417 417 32.0 769.0 89.0 1.0
# 30 370 417 417 32.0 2997.0 89.0 1.0

r = 370
s = 417
d = 417
p = (2*30+1) **2
rsd = r * s * d

xs = np.linspace(0, 1000)

ys = np.minimum(xs * 2.56e10, 1.792e11) # cpu
points = [ # cpu
    (1/12, 2 * rsd / 32.0 / 1000, 'pixel disparity'), # 32.0
    (1/16, rsd / 50.0 / 1000, 'prefix rows'), # 50.0
    (1/16, rsd / 28.5 / 1000, 'prefix cols'), # 28.5
    (1/8, rsd * p / 2997.0 / 1000, 'naive patch disparity'), # 
    (1/10, 4 * rsd / 95.0 / 1000, 'fast patch disparity'), # 95.0
    (1/8, 5 * rsd / 84.5 / 1000, 'costs'), # 84.5
    (0, 0 / 1.0 / 1000, 'traceback'), # 1.0
]


# 2.34552 14.106384 17.784304 9.156736 5.1363363 33.57387 0.57118404
# 2.390912 123.2245 33.747505 0.586176

ys = np.minimum(xs * 6.72e11, 5.098e11) # gpu
points = [ # gpu
    (1/12, 2 * rsd / 2.345 / 1000, 'pixel disparity'), # 32.0
    (1/16, rsd / 14.10 / 1000, 'prefix rows'), # 50.0
    (1/16, rsd / 17.8 / 1000, 'prefix cols'), # 28.5
    (1/8, rsd * p / 123.22 / 1000, 'naive patch disparity'), # 
    (1/10, 4 * rsd / 5.14 / 1000, 'fast patch disparity'), # 95.0
    (1/8, 5 * rsd / 33.574 / 1000, 'costs'), # 84.5
    (0, 0 / 0.5612 / 1000, 'traceback'), # 1.0
]

fig, ax = plt.subplots()
ax.set(xscale='log', yscale='log',  xlabel='arithmetic intensity (flops/byte)', ylabel='performance (flops/second)')
ax.plot(xs, ys)
for x, y, label in points:
    ax.scatter(x, y)
    ax.text(x, y, label, va='bottom', ha='left')
plt.show()




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
