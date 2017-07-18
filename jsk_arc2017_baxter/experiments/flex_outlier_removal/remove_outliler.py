#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt

from sklearn.covariance import EllipticEnvelope


data = np.loadtxt('./flex_bag_2017-07-18-21-42-53.csv', delimiter=',', skiprows=1, usecols=[0, 1])
time, flex = data[:, 0], data[:, 1]
time /= 1e9  # nsec -> sec
time -= time[0]  # align to t = 0

# input
X = flex[:, None]

# train
ratio_of_outlier = 0.2
clf = EllipticEnvelope(contamination=ratio_of_outlier)
clf.fit(X)

# estimate
is_inlier = clf.predict(X)
print(is_inlier)
print('Inlier: %d' % (is_inlier == 1).sum())
print('Outlier: %d' % (is_inlier == -1).sum())

# visualize
fig = plt.figure(figsize=(15, 7), tight_layout=True)

plt.subplot(121)
plt.plot(time, flex)
plt.xlabel('Time [s] (offset from t = 0)')
plt.ylabel('Flex sensor value')
plt.xlim(time.min(), time.max())
plt.ylim(X.min(), X.max())
plt.title('Flex values (before outlier removal)')

plt.subplot(122)
plt.plot(time[is_inlier == 1], X[is_inlier == 1])
plt.xlabel('Time [s] (offset from t = 0)')
plt.ylabel('Flex sensor value')
plt.xlim(time.min(), time.max())
plt.ylim(X.min(), X.max())
plt.title('Flex values (after outlier removal)')

# plt.show()
out_file = 'plot_remove_outlier.png'
plt.savefig(out_file)
print('Saved to: %s' % out_file)
