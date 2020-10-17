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
from chainercv.transforms.image.resize import resize
from chainercv.utils import non_maximum_suppression
import six


class OccludedMaskRCNN(chainer.Chain):

    def __init__(
            self, extractor, rpn, head, mean,
            min_size, max_size,
            loc_normalize_mean, loc_normalize_std,
    ):
        super(OccludedMaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.context = None
        self._detections_per_im = 100
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def prepare(self, img):
        _, H, W = img.shape

        scale = 1.
        scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)

        img = resize(img, (int(H * scale), int(W * scale)))
        img = (img - self.mean).astype(np.float32, copy=False)
        return img

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

    def predict(self, imgs):
        resized_imgs = []
        sizes = []
        scales = []
        for img in imgs:
            H, W = img.shape[1:]
            sizes.append((H, W))
            resized_img = self.prepare(img)
            scale = resized_img.shape[1] / H
            resized_imgs.append(resized_img)
            scales.append(scale)

        ins_labels, labels, bboxes, scores = [], [], [], []
        for resized_img, size, scale in zip(resized_imgs, sizes, scales):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(resized_img[None]))
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
                ins_labels.append(np.zeros((0, size[0], size[1]), dtype=bool))
                continue

            labels.append(label)
            bboxes.append(bbox)
            scores.append(score)

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
            roi_masks1, roi_masks2 = roi_masks
            roi_masks1 = roi_masks1.transpose((0, 2, 3, 1))
            roi_masks2 = roi_masks2.transpose((0, 2, 3, 1))
            roi_mask = cuda.to_cpu(roi_masks2.data)  # mainly used
            roi_mask_sub = cuda.to_cpu(roi_masks1.data)

            roi = np.round(bbox).astype(np.int32)
            n_roi = len(roi)
            ins_label = -1 * np.ones((n_roi, size[0], size[1]), dtype=np.int32)
            for i in six.moves.range(n_roi):
                y1, x1, y2, x2 = roi[i]
                roi_H = y2 - y1
                roi_W = x2 - x1
                roi_mask_i = cv2.resize(roi_mask[i], (roi_W, roi_H))
                roi_mask_sub_i = cv2.resize(
                    roi_mask_sub[i], (roi_W, roi_H))
                roi_mask_i = roi_mask_i + roi_mask_sub_i
                roi_mask_i_out = np.argmax(roi_mask_i, axis=2)
                ins_label[i, y1:y2, x1:x2] = roi_mask_i_out
            ins_labels.append(ins_label)

        return ins_labels, labels, bboxes, scores
