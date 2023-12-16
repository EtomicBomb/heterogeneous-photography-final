import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib as mpl

# # 32.0 49.0 28.0 26.0 93.0 83.0 1.0
# # pix  row  src   dst patch cost traceback

# 2.30272 13.125729 6.623232 10.830416 5.103312 33.16603 0.569648
# pixel,   rows,    src,     dst,      patch    costs    traceback

r = 370
s = 417
d = 417
p = 3
rsd = r * s * d

xs = np.logspace(-1.3, 1.5, 500)

stages = ['pixel divergence', 'prefix rows', 'prefix cols', 'naive patch divergence', 'fast patch divergence', 'costs']
ai = [2 * rsd /(r*s+r*d+rsd) / 8, 1/16, 1/16, ((2*p+1)**2 + 1)/16, 1/4, 1/5]
flops = [2 * rsd, rsd, rsd, rsd * ((2*p+1)**2 + 1), 4 * rsd, 5 * rsd]

print(len(stages), len(ai), len(flops))

title = 'flop rate for the cpu implementation'
ys = np.minimum(xs * 2.56e10, 1.792e11) # cpu
millis = [32.0, 50.0, 54, 477.5, 95.0, 84.5, 1.0] # cpu

title = 'flop rate for the gpu implementation'
ys = np.minimum(xs * 6.72e11, 5.098e11) # gpu
millis = [2.34552, 14.106384, 17.784304, 21.719, 5.1363363, 33.57387, 0.57118404] # gpu

# 

fig, ax = plt.subplots()
ax.set(xscale='log', yscale='log',  xlabel='arithmetic intensity (flops/byte)', ylabel='performance (flops/second)', title=title)
ax.plot(xs, ys)
for ai, flop, milli, stage in zip(ai, flops, millis, stages):
    ax.scatter(ai, flop / (milli / 1000), label=stage)
plt.legend(loc='upper left')
plt.show()
