import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib as mpl

r = 370
s = 417
d = 417
p = 30
rsd = r * s * d

xs = np.logspace(-1.3, 2.5, 500)

stages = ['pixel disparity', 'prefix rows', 'prefix cols', 'naive patch disparity', 'fast patch disparity', 'costs', 'traceback']
ai = [2 * rsd /(r*s+r*d+rsd) / 8, 1/16, 1/16, ((2*p+1)**2 + 1)/16, 1/4, 1/5]
flops = [2 * rsd, 2 * rsd, 2 * rsd, rsd * ((2*p+1)**2 + 1), 4 * rsd, 5 * rsd, 0]

# title = 'flop rate for the cpu implementation'
# ys = np.minimum(xs * 2.56e10, 1.792e11) # cpu
# millis = [32.0, 50.0, 28.5, 2997.0, 95.0, 84.5, 1.0] # cpu

title = 'flop rate for the gpu implementation'
ys = np.minimum(xs * 6.72e11, 5.098e11) # gpu
millis = [2.34552, 14.106384, 17.784304, 9.156736, 123.22, 5.1363363, 33.57387, 0.57118404] # gpu

fig, ax = plt.subplots()
ax.set(xscale='log', yscale='log',  xlabel='arithmetic intensity (flops/byte)', ylabel='performance (flops/second)', title=title)
ax.plot(xs, ys)
for ai, flop, milli, stage in zip(ai, flops, millis, stages):
    ax.scatter(ai, flop / (milli / 1000), label=stage)
plt.legend()
plt.show()




data = defaultdict(lambda: defaultdict(list))

stage_labels = ['pixel_similarity', 'sum_rows', 'sum_cols1', 'sum_cols2', 'patch_similarity', 'costs', 'traceback', 'patch_similarity2']
for line in open('reports/timings.txt', 'r').readlines():
    patch_size, rows, cols_src, cols_dst, *stage_times = map(float, line.strip().split(' '))
    for stage_time, stage_label in zip(stage_times, stage_labels):
        data[stage_label][patch_size].append(stage_time)

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
        pixel, time = zip(*pixel_time)

        print(pixel, time)
        bar_container = ax.barh(x + offset, list(time), width, label=f'patch_size={patch_size}')
        ax.bar_label(bar_container, fmt='{:,.2f}')
        #ax.plot(pixel, time, label=f'patch_size={patch_size}')
        multiplier += 1

    #ax.set_yticks(x+width,  ['small', 'medium', 'large'])
    ax.set_yticks(x+width,  stage_labels)
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
