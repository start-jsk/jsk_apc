from __future__ import division

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator

from grasp_data_generator.models.occluded_grasp_mask_rcnn.utils \
    import ProposalTargetCreator


class OccludedGraspMaskRCNNTrainChain(chainer.Chain):

    def __init__(
            self, occluded_grasp_mask_rcnn, rpn_sigma=3., roi_sigma=1.,
            anchor_target_creator=AnchorTargetCreator(),
            proposal_target_creator=ProposalTargetCreator(),
            grasp_branch_finetune=False, alpha_graspable=0.05,
            mask_branch_finetune=False,
    ):
        super(OccludedGraspMaskRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.occluded_grasp_mask_rcnn = occluded_grasp_mask_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator
        self.grasp_branch_finetune = grasp_branch_finetune
        self.alpha_graspable = alpha_graspable
        self.mask_branch_finetune = mask_branch_finetune
        assert not (mask_branch_finetune and grasp_branch_finetune)

        self.loc_normalize_mean = occluded_grasp_mask_rcnn.loc_normalize_mean
        self.loc_normalize_std = occluded_grasp_mask_rcnn.loc_normalize_std

    def __call__(
            self, imgs, ins_labels, labels, bboxes, scales,
            sg_masks=None, dg_masks=None, rotations=None,
    ):
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scales, chainer.Variable):
            scales = scales.data
        n = len(bboxes)
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        scale = cuda.to_cpu(scales)[0]

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.occluded_grasp_mask_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.occluded_grasp_mask_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        ins_label = ins_labels[0]
        sg_mask = None if sg_masks is None else sg_masks[0]
        dg_mask = None if dg_masks is None else dg_masks[0]
        rotation = None if rotations is None else rotations[0]

        if len(bbox) == 0:
            return chainer.Variable(self.xp.array(0, dtype=np.float32))

        # Sample RoIs and forward
        target = self.proposal_target_creator(
            roi, bbox, label, ins_label, sg_mask, dg_mask, rotation,
            loc_normalize_mean=self.loc_normalize_mean,
            loc_normalize_std=self.loc_normalize_std,
            rotate_angle=self.occluded_grasp_mask_rcnn.rotate_angle)
        sample_roi, gt_roi_loc, gt_roi_label = target[:3]
        gt_roi_mask, gt_roi_sg_mask, gt_roi_dg_mask = target[3:]

        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)

        roi_cls_loc, roi_score, roi_mask, roi_sg_mask, roi_dg_mask = \
            self.occluded_grasp_mask_rcnn.head(
                features, sample_roi, sample_roi_index, labels=gt_roi_label,
                pred_bbox=True, pred_mask=True,
                pred_bbox2=False, pred_mask2=True,
                grasp_branch_finetune=self.grasp_branch_finetune,
                mask_branch_finetune=self.mask_branch_finetune)

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

        if self.mask_branch_finetune:
            roi_sg_mask_loss = chainer.Variable(
                self.xp.array(0, dtype=np.float32))
            roi_dg_mask_loss = chainer.Variable(
                self.xp.array(0, dtype=np.float32))
        else:
            # loss for grasp mask
            roi_sg_mask1, roi_sg_mask2 = roi_sg_mask
            sg_frq = np.bincount(
                cuda.to_cpu(gt_roi_sg_mask[:n_positive]).flatten(),
                minlength=self.occluded_grasp_mask_rcnn.head.n_grasp_class)
            sg_frq = sg_frq.astype(np.float32)
            if sum(sg_frq[1:]) == 0:
                sg_weight = np.zeros((len(sg_frq),), dtype=np.float32)
                sg_weight[0] = 1.0
            elif sg_frq[0] == 0:
                sg_weight = np.zeros((len(sg_frq),), dtype=np.float32)
                sg_weight[np.where(sg_frq > 0)[0][0]] = 1.0
            else:
                sg_weight = sg_frq.copy()
                sg_weight[sg_frq == 0] = np.inf
                sg_weight = sum(sg_frq) / sg_weight
                sg_weight = sg_weight / sum(sg_weight)

            if self.grasp_branch_finetune:
                sg_weight[1:] = sg_weight[1:] / self.alpha_graspable

            if self.xp != np:
                sg_weight = cuda.to_gpu(sg_weight)
            roi_sg_mask_loss = F.softmax_cross_entropy(
                roi_sg_mask1, gt_roi_sg_mask[:n_positive],
                class_weight=sg_weight)
            # mask2 = mask1 + mask2
            roi_sg_mask2 = roi_sg_mask1 + roi_sg_mask2
            roi_sg_mask_loss += F.softmax_cross_entropy(
                roi_sg_mask2, gt_roi_sg_mask[:n_positive],
                class_weight=sg_weight)

            roi_dg_mask1, roi_dg_mask2 = roi_dg_mask
            dg_frq = np.bincount(
                cuda.to_cpu(gt_roi_dg_mask[:n_positive]).flatten(),
                minlength=self.occluded_grasp_mask_rcnn.head.n_grasp_class)
            dg_frq = dg_frq.astype(np.float32)
            if sum(dg_frq[1:]) == 0:
                dg_weight = np.zeros((len(dg_frq),), dtype=np.float32)
                dg_weight[0] = 1.0
            elif dg_frq[0] == 0:
                dg_weight = np.zeros((len(dg_frq),), dtype=np.float32)
                dg_weight[np.where(dg_frq > 0)[0][0]] = 1.0
            else:
                dg_weight = dg_frq.copy()
                dg_weight[dg_frq == 0] = np.inf
                dg_weight = sum(dg_frq) / dg_weight
                dg_weight = dg_weight / sum(dg_weight)

            if self.grasp_branch_finetune:
                dg_weight[1:] = dg_weight[1:] / self.alpha_graspable

            if self.xp != np:
                dg_weight = cuda.to_gpu(dg_weight)
            roi_dg_mask_loss = F.softmax_cross_entropy(
                roi_dg_mask1, gt_roi_dg_mask[:n_positive],
                class_weight=dg_weight)
            # mask2 = mask1 + mask2
            roi_dg_mask2 = roi_dg_mask1 + roi_dg_mask2
            roi_dg_mask_loss += F.softmax_cross_entropy(
                roi_dg_mask2, gt_roi_dg_mask[:n_positive],
                class_weight=dg_weight)

        # # DEBUG
        # print('sg_mask: gt')
        # print(np.bincount(cuda.to_cpu(gt_roi_sg_mask[:n_positive]).flatten()))
        # print('sg_mask: predict')
        # print(np.bincount(np.argmax(
        #     cuda.to_cpu(roi_sg_mask1.array), axis=1).flatten()))
        # print(np.bincount(np.argmax(
        #     cuda.to_cpu(roi_sg_mask2.array), axis=1).flatten()))
        # print('dg_mask: gt')
        # print(np.bincount(cuda.to_cpu(gt_roi_dg_mask[:n_positive]).flatten()))
        # print('dg_mask: predict')
        # print(np.bincount(np.argmax(
        #     cuda.to_cpu(roi_dg_mask1.array), axis=1).flatten()))
        # print(np.bincount(np.argmax(
        #     cuda.to_cpu(roi_dg_mask2.array), axis=1).flatten()))
        # # DEBUG

        if self.grasp_branch_finetune:
            loss = roi_sg_mask_loss + roi_dg_mask_loss
        elif self.mask_branch_finetune:
            loss = rpn_loc_loss + rpn_cls_loss
            loss = loss + roi_loc_loss + roi_cls_loss + roi_mask_loss
        else:
            loss = rpn_loc_loss + rpn_cls_loss
            loss = loss + roi_loc_loss + roi_cls_loss + roi_mask_loss
            loss = loss + roi_sg_mask_loss + roi_dg_mask_loss

        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'roi_loc_loss': roi_loc_loss,
            'roi_cls_loss': roi_cls_loss,
            'roi_mask_loss': roi_mask_loss,
            'roi_sg_mask_loss': roi_sg_mask_loss,
            'roi_dg_mask_loss': roi_dg_mask_loss,
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
