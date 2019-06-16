from __future__ import division

import numpy as np
import six

from chainer_mask_rcnn.utils import get_mask_overlap


def eval_sigmoid_segmentation(pred_labels, gt_labels, channel_names=None):

    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    iou_all = {}
    out_channels = None
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        assert pred_label.ndim == 3
        assert gt_label.ndim == 3
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')

        out_channels = pred_label.shape[0]

        for c in range(out_channels):
            if c not in iou_all:
                iou_all[c] = []

            keep = gt_label[c, :, :] >= 0
            if keep.sum() == 0:
                iou_all[c].append([np.nan, np.nan])
                continue

            iou_ci = []
            for m in [0, 1]:
                pred_label_c = pred_label[c, :, :][keep] == m
                gt_label_c = gt_label[c, :, :][keep] == m
                if gt_label_c.sum() == 0:
                    iou_cim = np.nan
                else:
                    iou_cim = get_mask_overlap(pred_label_c, gt_label_c)
                iou_ci.append(iou_cim)
            iou_all[c].append(iou_ci)

    n_sample = len(iou_all[0])
    iou = np.full((out_channels, 2), np.nan, dtype=float)
    for c in range(out_channels):
        assert len(iou_all[c]) == n_sample
        iou_c = np.nanmean(iou_all[c], axis=0)
        iou[c] = iou_c
    iou = np.nanmean(iou, axis=1)

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    result = {'iou': iou, 'miou': np.nanmean(iou)}

    if channel_names is not None:
        assert len(channel_names) == out_channels
        for channel_name, iou_c in zip(channel_names, iou):
            result['iou/%s' % channel_name] = iou_c

    return result
