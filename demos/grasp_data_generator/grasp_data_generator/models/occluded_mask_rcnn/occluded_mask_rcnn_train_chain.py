import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainer_mask_rcnn.models.utils import ProposalTargetCreator
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator


class OccludedMaskRCNNTrainChain(chainer.Chain):

    def __init__(self, occluded_mask_rcnn, rpn_sigma=3., roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator()):
        super(OccludedMaskRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.occluded_mask_rcnn = occluded_mask_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = occluded_mask_rcnn.loc_normalize_mean
        self.loc_normalize_std = occluded_mask_rcnn.loc_normalize_std

    def __call__(self, imgs, ins_labels, labels, bboxes, scale):
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        scale = np.asscalar(cuda.to_cpu(scale))
        n = len(bboxes)
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.occluded_mask_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.occluded_mask_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        ins_label = ins_labels[0]

        if len(bbox) == 0:
            return chainer.Variable(self.xp.array(0, dtype=np.float32))

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask = \
            self.proposal_target_creator(roi, bbox, label, ins_label)

        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)

        roi_cls_loc, roi_score, roi_mask = self.occluded_mask_rcnn.head(
            features, sample_roi, sample_roi_index, labels=gt_roi_label,
            pred_bbox=True, pred_mask=True,
            pred_bbox2=False, pred_mask2=True)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        if not isinstance(roi_cls_loc, tuple):
            roi_cls_loc = (roi_cls_loc,)
        if not isinstance(roi_score, tuple):
            roi_score = (roi_score,)

        roi_loc_loss = 0
        roi_cls_loss = 0
        for roi_cls_loc_i, roi_score_i in zip(roi_cls_loc, roi_score):
            # Losses for outputs of the head.
            n_sample = roi_cls_loc_i.shape[0]
            roi_cls_loc_i = roi_cls_loc_i.reshape((n_sample, -1, 4))
            roi_loc = roi_cls_loc_i[self.xp.arange(n_sample), gt_roi_label]
            roi_loc_loss += _fast_rcnn_loc_loss(
                roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
            roi_cls_loss += F.softmax_cross_entropy(roi_score_i, gt_roi_label)

        # Losses for outputs of mask branch
        roi_mask1, roi_mask2 = roi_mask
        n_positive = int((gt_roi_label > 0).sum())
        roi_mask_loss = F.softmax_cross_entropy(
            roi_mask1, gt_roi_mask[:n_positive])
        roi_mask2 = roi_mask1 + roi_mask2
        roi_mask_loss += F.softmax_cross_entropy(
            roi_mask2, gt_roi_mask[:n_positive])

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + \
            roi_mask_loss
        chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'roi_loc_loss': roi_loc_loss,
                                 'roi_cls_loss': roi_cls_loss,
                                 'roi_mask_loss': roi_mask_loss,
                                 'loss': loss},
                                self)
        return loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    return F.sum(y)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)

    in_weight = xp.zeros_like(gt_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= xp.sum(gt_label >= 0)
    return loc_loss
