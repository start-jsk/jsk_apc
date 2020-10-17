# Modified works:
# --------------------------------------------------------
# Copyright (c) 2017 - 2018 Kentaro Wada.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Original works:
# --------------------------------------------------------
# Copyright (c) 2017 Shingo Kitagawa.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/knorth55/chainer-fcis
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/chainer/chainercv
# --------------------------------------------------------

from chainer import cuda
from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.utils.bbox.bbox_iou import bbox_iou
import cv2
import numpy as np


def rot_to_rot_lbl(rot, rotate_angle):
    rot_lbl = int((90 - rot) / rotate_angle)
    if ((90 - rot) % rotate_angle) > (rotate_angle / 2.0):
        rot_lbl += 1
    # [background, 90, 90 - rotate_angle....]
    rot_lbl = rot_lbl % int(180 // rotate_angle) + 1
    return rot_lbl


def rot_lbl_to_rot(rot_lbl, rotate_angle):
    if rot_lbl == 0:
        return None
    rot = 90 - (rot_lbl - 1) * rotate_angle
    return rot


class ProposalTargetCreator(object):

    def __init__(
            self,
            n_sample=512,
            pos_ratio=0.25, pos_iou_thresh=0.5,
            neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
            mask_size=14, binary_thresh=0.4,
    ):

        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

    def __call__(
            self, roi, bbox, label, mask, sg_mask, dg_mask, rotation,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            rotate_angle=None,
    ):

        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)
        sg_mask = cuda.to_cpu(sg_mask)
        dg_mask = cuda.to_cpu(dg_mask)
        rotation = cuda.to_cpu(rotation)

        n_bbox, _ = bbox.shape
        if n_bbox == 0:
            raise ValueError('Empty bbox is not supported.')

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        gt_roi_mask = -1 * np.ones(
            (len(sample_roi), self.mask_size, self.mask_size),
            dtype=np.int32)
        if sg_mask is not None:
            gt_roi_sg_mask = -1 * np.ones(
                (len(sample_roi), self.mask_size, self.mask_size),
                dtype=np.int32)
        if dg_mask is not None:
            gt_roi_dg_mask = -1 * np.ones(
                (len(sample_roi), self.mask_size, self.mask_size),
                dtype=np.int32)

        for i, pos_ind in enumerate(pos_index):
            bb = np.round(sample_roi[i]).astype(np.int32)
            # Compute gt masks
            gt_mask = mask[gt_assignment[pos_ind]]
            gt_roi_mask_i = gt_mask[bb[0]:bb[2], bb[1]:bb[3]]
            gt_roi_mask_i_score = (
                np.arange(gt_roi_mask_i.max() + 1) ==
                gt_roi_mask_i[..., None]).astype(np.float32)  # label -> onehot
            gt_roi_mask_i_score = cv2.resize(
                gt_roi_mask_i_score, (self.mask_size, self.mask_size))
            if gt_roi_mask_i_score.ndim == 2:
                gt_roi_mask_i_score = gt_roi_mask_i_score.reshape(
                    gt_roi_mask_i_score.shape[:2] + (1,))
            gt_roi_mask[i] = np.argmax(
                gt_roi_mask_i_score, axis=2).astype(np.int32)

            if sg_mask is not None:
                # Compute sg mask
                if not (rotation is None or rotate_angle is None):
                    gt_rot = rotation[gt_assignment[pos_ind]]
                    gt_rot_lbl = rot_to_rot_lbl(gt_rot, rotate_angle)
                gt_sg_msk = sg_mask[gt_assignment[pos_ind]]
                gt_roi_sg_msk = gt_sg_msk[bb[0]:bb[2], bb[1]:bb[3]]
                # use only visible region
                gt_roi_sg_msk = cv2.resize(
                    gt_roi_sg_msk.astype(np.float32),
                    (self.mask_size, self.mask_size))
                gt_roi_sg_msk = (gt_roi_sg_msk >= self.binary_thresh)
                gt_roi_sg_msk = gt_roi_sg_msk.astype(np.int32)
                if rotate_angle is not None:
                    gt_roi_sg_msk = gt_roi_sg_msk * gt_rot_lbl
                gt_roi_sg_mask[i] = gt_roi_sg_msk

            if dg_mask is not None:
                # Compute dg mask
                gt_dg_msk = dg_mask[gt_assignment[pos_ind]]
                gt_roi_dg_msk = gt_dg_msk[bb[0]:bb[2], bb[1]:bb[3]]
                # use only visible region
                gt_roi_dg_msk[gt_roi_mask_i != 1] = False
                gt_roi_dg_msk = cv2.resize(
                    gt_roi_dg_msk.astype(np.float32),
                    (self.mask_size, self.mask_size))
                gt_roi_dg_msk = (gt_roi_dg_msk >= self.binary_thresh)
                gt_roi_dg_msk = gt_roi_dg_msk.astype(np.int32)
                if rotate_angle is not None:
                    gt_roi_dg_msk = gt_roi_dg_msk * gt_rot_lbl
                gt_roi_dg_mask[i] = gt_roi_dg_msk

        gt_roi_sg_mask = None if sg_mask is None else gt_roi_sg_mask
        gt_roi_dg_mask = None if dg_mask is None else gt_roi_dg_mask
        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_loc = cuda.to_gpu(gt_roi_loc)
            gt_roi_label = cuda.to_gpu(gt_roi_label)
            gt_roi_mask = cuda.to_gpu(gt_roi_mask)
            gt_roi_sg_mask = cuda.to_gpu(gt_roi_sg_mask)
            gt_roi_dg_mask = cuda.to_gpu(gt_roi_dg_mask)
        return sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask, \
            gt_roi_sg_mask, gt_roi_dg_mask
