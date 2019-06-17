# Mofidied work:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# --------------------------------------------------------
#
# Original works by:
# --------------------------------------------------------
# Faster R-CNN implementation by Chainer
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/mitmul/chainer-faster-rcnn
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

from __future__ import division

import cv2
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression
import six


class MaskRCNN(chainer.Chain):

    mask_losses = [
        'softmax',
        'softmax_x2',
        'sigmoid_softmax',
        'sigmoid_sigmoid',
        'sigmoid_sigmoid+',
        'softmax_relook_softmax',
        'softmax_relook_softmax+',
        'softmax_relook_softmax+_res',
        'softmax_relook_softmax_cls',
        'softmax_relook_softmax+_cls',
        'softmax_relook_softmax_tt',
        'softmax_relook_softmax+_tt',
        'softmax_relook_softmax+_tt2',
        'softmax_relook_softmax_cls_tt',
        'softmax_relook_softmax+_cls_tt',
        'softmax_relook_softmax_bbox',
        'softmax_relook_softmax+_bbox',
    ]

    def __init__(
            self, extractor, rpn, head,
            mean,
            min_size=600,
            max_size=1000,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            detections_per_im=100,
            mask_loss='softmax'):
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self._detections_per_im = detections_per_im

        assert mask_loss in self.mask_losses
        self.mask_loss = mask_loss

        self.context = None
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def prepare(self, imgs):
        prepared_imgs = []
        sizes = []
        scales = []
        for img in imgs:
            _, H, W = img.shape

            scale = 1.

            if self.min_size:
                scale = self.min_size / min(H, W)

            if self.max_size and scale * max(H, W) > self.max_size:
                scale = self.max_size / max(H, W)

            img = img.transpose(1, 2, 0)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            img = img.transpose(2, 0, 1)

            img = (img - self.mean).astype(np.float32, copy=False)

            prepared_imgs.append(img)
            sizes.append((H, W))
            scales.append(scale)
        return prepared_imgs, sizes, scales

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]

            # thresholding by score
            keep = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[keep]
            prob_l = prob_l[keep]

            # thresholding by nms
            keep = non_maximum_suppression(
                cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def _to_bbox_label_score(self, roi_cls_locs, roi_scores, rois, roi_indices, scale, size):  # NOQA
        # We are assuming that batch size is 1.
        roi_cls_loc = roi_cls_locs.data
        roi_score = roi_scores.data
        roi = rois / scale
        roi_index = roi_indices

        # Convert predictions to bounding boxes in image coordinates.
        # Bounding boxes are scaled to the scale of the input images.
        mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean),
                            self.n_class)
        std = self.xp.tile(self.xp.asarray(self.loc_normalize_std),
                           self.n_class)
        roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
        roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
        roi_cls = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
        cls_bbox = loc2bbox(roi_cls.reshape((-1, 4)),
                            roi_cls_loc.reshape((-1, 4)))
        cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
        # clip bounding box
        cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, size[0])
        cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, size[1])
        # clip roi
        roi[:, 0::2] = self.xp.clip(roi[:, 0::2], 0, size[0])
        roi[:, 1::2] = self.xp.clip(roi[:, 1::2], 0, size[1])

        prob = F.softmax(roi_score).data

        roi_index = self.xp.broadcast_to(
            roi_index[:, None], roi_cls_loc.shape[:2])
        raw_cls_bbox = cuda.to_cpu(cls_bbox)
        raw_prob = cuda.to_cpu(prob)

        if self.context:
            n_fg_class = self.n_class - 1
            for l in range(n_fg_class):
                if l not in self.context:
                    raw_prob[:, l + 1] = 0
            raw_prob = raw_prob / raw_prob.sum(axis=0)
        bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)

        bbox_int = np.round(bbox).astype(np.int32)
        bbox_sizes = ((bbox_int[:, 2] - bbox_int[:, 0]) *
                      (bbox_int[:, 3] - bbox_int[:, 1]))
        keep = bbox_sizes > 0
        bbox = bbox[keep]
        label = label[keep]
        score = score[keep]

        if self._detections_per_im > 0:
            indices = np.argsort(score)
            keep = indices >= (len(indices) - self._detections_per_im)
            bbox = bbox[keep]
            label = label[keep]
            score = score[keep]

        return bbox, label, score

    def predict_masks(self, imgs):
        return self.predict(imgs)

    def predict(self, imgs):
        imgs, sizes, scales = self.prepare(imgs)

        bboxes = list()
        masks = list()
        labels = list()
        scores = list()
        for img, size, scale in zip(imgs, sizes, scales):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[None]))
                h = self.extractor(x)
                rpn_locs, rpn_scores, rois, roi_indices, anchor =\
                    self.rpn(h, x.shape[2:], scale)
                roi_cls_locs, roi_scores, _, = self.head(
                    h, rois, roi_indices,
                    pred_bbox=True, pred_mask=False,
                    pred_bbox2=False, pred_mask2=False)

            bbox, label, score = self._to_bbox_label_score(
                roi_cls_locs, roi_scores, rois, roi_indices, scale, size)

            if len(bbox) == 0:
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)
                masks.append(np.zeros((0, size[0], size[1]), dtype=bool))
                continue

            if self.mask_loss in [
                'softmax_relook_softmax_bbox',
                'softmax_relook_softmax+_bbox',
            ]:
                with chainer.using_config('train', False), \
                        chainer.function.no_backprop_mode():
                    rois = self.xp.asarray(bbox) * scale
                    roi_indices = self.xp.zeros(
                        (len(bbox),), dtype=np.int32)
                    roi_cls_locs, roi_scores, _, = self.head(
                        h, rois, roi_indices,
                        pred_bbox=False, pred_mask=True,
                        pred_bbox2=True, pred_mask2=False,
                        labels=label + 1)
                    roi_cls_locs = roi_cls_locs[1]
                    roi_scores = roi_scores[1]
                bbox, label, score = self._to_bbox_label_score(
                    roi_cls_locs, roi_scores, rois, roi_indices, scale, size)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

            if len(bbox) == 0:
                masks.append(np.zeros((0, size[0], size[1]), dtype=bool))
                continue

            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                rois = self.xp.asarray(bbox) * scale
                roi_indices = self.xp.zeros(
                    (len(bbox),), dtype=np.int32)
                _, _, roi_masks = self.head(
                    x=h, rois=rois, roi_indices=roi_indices,
                    pred_bbox=False, pred_mask=True,
                    pred_bbox2=False, pred_mask2=True,
                    labels=label + 1)
            if self.mask_loss in ['softmax', 'softmax_x2']:
                _, n_fg_class_x3, roi_H, roi_W = roi_masks.shape
                n_fg_class = n_fg_class_x3 // 3
                roi_masks = roi_masks.reshape(
                    (roi_masks.shape[0], 3, n_fg_class, roi_H, roi_W))
                roi_masks = roi_masks[np.arange(len(label)), :, label, :, :]
                roi_masks = roi_masks.transpose(0, 2, 3, 1)
                roi_mask = cuda.to_cpu(roi_masks.data)
            elif self.mask_loss == 'sigmoid_softmax':
                roi_masks, roi_masks_bginv = roi_masks
                # roi_mask
                roi_masks = F.sigmoid(roi_masks)
                roi_masks = roi_masks[np.arange(len(label)), label, :, :]
                roi_mask = cuda.to_cpu(roi_masks.data)
                # roi_mask_bginv
                _, n_fg_class_x2, roi_H, roi_W = roi_masks_bginv.shape
                n_fg_class = n_fg_class_x2 // 2
                roi_masks_bginv = roi_masks_bginv.reshape(
                    (roi_masks_bginv.shape[0], 2, n_fg_class, roi_H, roi_W))
                roi_masks_bginv = roi_masks_bginv[
                    np.arange(len(label)), :, label, :, :]
                roi_masks_bginv = F.softmax(roi_masks_bginv)[:, 1, :, :]
                roi_mask_bginv = cuda.to_cpu(roi_masks_bginv.data)
            elif self.mask_loss in ['sigmoid_sigmoid', 'sigmoid_sigmoid+']:
                roi_masks, roi_masks_bginv = roi_masks
                # roi_mask
                roi_masks = F.sigmoid(roi_masks)
                roi_masks = roi_masks[np.arange(len(label)), label, :, :]
                roi_mask = cuda.to_cpu(roi_masks.data)
                # roi_mask_bginv
                roi_masks_bginv = F.sigmoid(roi_masks_bginv)
                roi_masks_bginv = roi_masks_bginv[
                    np.arange(len(label)), label, :, :]
                roi_mask_bginv = cuda.to_cpu(roi_masks_bginv.data)
            elif self.mask_loss in [
                'softmax_relook_softmax',
                'softmax_relook_softmax+',
                'softmax_relook_softmax+_res',
                'softmax_relook_softmax_cls',
                'softmax_relook_softmax+_cls',
                'softmax_relook_softmax_bbox',
                'softmax_relook_softmax+_bbox',
                'softmax_relook_softmax_tt',
                'softmax_relook_softmax+_tt',
                'softmax_relook_softmax+_tt2',
                'softmax_relook_softmax_cls_tt',
                'softmax_relook_softmax+_cls_tt',
            ]:
                roi_masks1, roi_masks2 = roi_masks
                roi_masks1 = roi_masks1.transpose(0, 2, 3, 1)
                roi_masks2 = roi_masks2.transpose(0, 2, 3, 1)
                roi_mask = cuda.to_cpu(roi_masks2.data)  # mainly used
                roi_mask_sub = cuda.to_cpu(roi_masks1.data)
            else:
                raise ValueError

            roi = np.round(bbox).astype(np.int32)
            n_roi = len(roi)
            mask = - np.ones((n_roi, size[0], size[1]), dtype=np.int32)
            for i in six.moves.range(n_roi):
                y1, x1, y2, x2 = roi[i]
                roi_H = y2 - y1
                roi_W = x2 - x1
                roi_mask_i = cv2.resize(roi_mask[i], (roi_W, roi_H))
                if self.mask_loss in ['softmax', 'softmax_x2']:
                    roi_mask_i_out = np.argmax(roi_mask_i, axis=2)
                elif self.mask_loss in [
                    'softmax_relook_softmax',
                    'softmax_relook_softmax+',
                    'softmax_relook_softmax+_res',
                    'softmax_relook_softmax_cls',
                    'softmax_relook_softmax+_cls',
                    'softmax_relook_softmax_bbox',
                    'softmax_relook_softmax+_bbox',
                    'softmax_relook_softmax_tt',
                    'softmax_relook_softmax+_tt',
                    'softmax_relook_softmax+_tt2',
                    'softmax_relook_softmax_cls_tt',
                    'softmax_relook_softmax+_cls_tt',
                ]:
                    if '+' in self.mask_loss:
                        roi_mask_sub_i = cv2.resize(
                            roi_mask_sub[i], (roi_W, roi_H))
                        roi_mask_i = roi_mask_i + roi_mask_sub_i
                    roi_mask_i_out = np.argmax(roi_mask_i, axis=2)
                elif self.mask_loss in [
                    'sigmoid_softmax',
                    'sigmoid_sigmoid',
                    'sigmoid_sigmoid+',
                ]:
                    roi_mask_bginv_i = cv2.resize(
                        roi_mask_bginv[i], (roi_W, roi_H))

                    roi_mask_i_out = - np.ones_like(roi_mask_i)
                    roi_mask_vis_i = roi_mask_i > 0.5
                    roi_mask_i_out[roi_mask_vis_i] = 1  # vis
                    roi_mask_i_out[~roi_mask_vis_i &
                                   (roi_mask_bginv_i <= 0.5)] = 0   # bg
                    roi_mask_i_out[~roi_mask_vis_i &
                                   (roi_mask_bginv_i > 0.5)] = 2   # inv
                else:
                    raise ValueError
                mask[i, y1:y2, x1:x2] = roi_mask_i_out
            masks.append(mask)

        return bboxes, masks, labels, scores
