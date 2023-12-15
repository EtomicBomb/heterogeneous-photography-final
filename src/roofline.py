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
