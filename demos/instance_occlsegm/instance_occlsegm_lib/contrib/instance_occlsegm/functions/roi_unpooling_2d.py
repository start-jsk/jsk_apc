#!/usr/bin/env python

import numpy as np

from chainer.backends import cuda
from chainer import Function
from chainer.functions.array.resize_images import ResizeImages
from chainer.functions.array.resize_images import ResizeImagesGrad
from chainer.utils import type_check


class ROIUnpooling2D(Function):

    def __init__(self, outb, outh, outw, spatial_scale):
        self.outb = outb
        self.outh = outh
        self.outw = outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim == 4,
            roi_type.dtype == np.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        x, rois = inputs
        self.retain_inputs((1,))
        self._x_shape = x.shape
        assert x.shape[0] == rois.shape[0]
        y = xp.zeros(
            (self.outb, x.shape[1], self.outh, self.outw), dtype=x.dtype
        )

        rois = cuda.to_cpu(rois)
        n_roi = len(rois)
        for i in range(n_roi):
            xi = x[i]
            roi = rois[i]

            batch_ind, x1, y1, x2, y2 = roi
            batch_ind = int(batch_ind)
            x1 = min(self.outw, max(0, int(round(x1 * self.spatial_scale))))
            x2 = min(self.outw, max(0, int(round(x2 * self.spatial_scale))))
            y1 = min(self.outh, max(0, int(round(y1 * self.spatial_scale))))
            y2 = min(self.outh, max(0, int(round(y2 * self.spatial_scale))))

            roih, roiw = y2 - y1, x2 - x1
            yi = ResizeImages((roih, roiw)).apply((xi[None],))[0][0].array
            y[batch_ind, :, y1:y2, x1:x2] += yi

        return y,

    def backward(self, inputs, gys):
        xp = cuda.get_array_module(*inputs)
        _, rois = inputs
        gy, = gys
        gx = xp.zeros(self._x_shape, dtype=gy.dtype)

        B, C, H, W = self._x_shape

        rois = cuda.to_cpu(rois)
        rois = np.round(rois).astype(np.int32)
        pooledh, pooledw = self._x_shape[2:4]
        for i in range(len(rois)):
            roi = rois[i]

            batch_ind, x1, y1, x2, y2 = roi
            batch_ind = int(batch_ind)
            x1 = min(self.outw, max(0, int(round(x1 * self.spatial_scale))))
            x2 = min(self.outw, max(0, int(round(x2 * self.spatial_scale))))
            y1 = min(self.outh, max(0, int(round(y1 * self.spatial_scale))))
            y2 = min(self.outh, max(0, int(round(y2 * self.spatial_scale))))

            roih, roiw = y2 - y1, x2 - x1
            gyi = gy[batch_ind, :, y1:y2, x1:x2]
            gxi = ResizeImagesGrad(
                input_shape=(1, C, H, W),
                output_shape=(roih, roiw)
            ).apply((gyi[None],))[0][0].array

            gx[i, :] = gxi
        return gx, None


def roi_unpooling_2d(x, rois, outb, outh, outw, spatial_scale, axes='xy'):
    if axes not in ['xy', 'yx']:
        raise ValueError('Unsupported axes: {}'.format(axes))
    if axes == 'yx':
        rois = rois[:, [0, 2, 1, 4, 3]]
    return ROIUnpooling2D(outb, outh, outw, spatial_scale)(x, rois)
