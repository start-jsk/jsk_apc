import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainer_mask_rcnn.models.utils import ProposalTargetCreator
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator


class MaskRCNNTrainChain(chainer.Chain):

    """Calculate losses for Faster R-CNN and report them.

    This is used to train Faster R-CNN in the joint training scheme
    [#FRCNN]_.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    .. [#FRCNN] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        faster_rcnn (~chainercv.links.model.faster_rcnn.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#FRCNN]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#FRCNN]_.
        anchor_target_creator: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator_params: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.ProposalTargetCreator`.

    """

    def __init__(self, mask_rcnn, rpn_sigma=3., roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator()):
        super(MaskRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.mask_rcnn = mask_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = mask_rcnn.loc_normalize_mean
        self.loc_normalize_std = mask_rcnn.loc_normalize_std

    def __call__(self, imgs, bboxes, labels, masks, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
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

        features = self.mask_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.mask_rcnn.rpn(
            features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        mask = masks[0]

        if len(bbox) == 0:
            return chainer.Variable(self.xp.array(0, dtype=np.float32))

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask = \
            self.proposal_target_creator(roi, bbox, label, mask)

        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)

        if self.mask_rcnn.mask_loss in ['softmax_relook_softmax_tt',
                                        'softmax_relook_softmax+_tt',
                                        'softmax_relook_softmax+_tt2',
                                        'softmax_relook_softmax_cls_tt',
                                        'softmax_relook_softmax+_cls_tt']:
            roi_cls_loc, roi_score, roi_mask1 = self.mask_rcnn.head(
                features, sample_roi, sample_roi_index,
                pred_bbox=True, pred_mask=True,
                pred_bbox2=False, pred_mask2=True, labels=gt_roi_label)
            # size: original, img_size: current
            size = (img_size[0] / scale, img_size[1] / scale)
            roi2, _, _ = self.mask_rcnn._to_bbox_label_score(
                roi_cls_loc, roi_score, sample_roi, sample_roi_index,
                scale, size)
            roi2 = roi2 * scale  # original -> current img_size
            # it overwrites gt_roi_mask
            if self.xp != np:
                roi2 = cuda.to_gpu(roi2)
            sample_roi2, _, gt_roi_label2, gt_roi_mask2 = \
                self.proposal_target_creator(roi2, bbox, label, mask)
            sample_roi_index2 = self.xp.zeros(
                (len(sample_roi2),), dtype=np.int32)
            _, _, roi_mask2 = self.mask_rcnn.head(
                features, sample_roi2, sample_roi_index2,
                pred_bbox=False, pred_mask=True,
                pred_bbox2=False, pred_mask2=True,
                labels=gt_roi_label2)
            roi_mask = (roi_mask1, roi_mask2)
        else:
            pred_bbox2 = self.mask_rcnn.mask_loss in [
                'softmax_relook_softmax_bbox',
                'softmax_relook_softmax+_bbox',
            ]
            roi_cls_loc, roi_score, roi_mask = self.mask_rcnn.head(
                features, sample_roi, sample_roi_index, labels=gt_roi_label,
                pred_bbox=True, pred_mask=True,
                pred_bbox2=pred_bbox2, pred_mask2=True)

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
        if self.mask_rcnn.mask_loss in ['softmax', 'softmax_x2']:
            assert roi_mask.shape[0] == n_sample
            _, n_fg_class_x3, roi_H, roi_W = roi_mask.shape
            n_fg_class = n_fg_class_x3 // 3
            roi_mask = F.reshape(
                roi_mask, (n_sample, 3, n_fg_class, roi_H, roi_W))
            assert gt_roi_mask.shape == (n_sample, roi_H, roi_W)
            roi_mask_loss = F.softmax_cross_entropy(
                roi_mask[np.arange(n_sample), :, gt_roi_label - 1, :, :],
                gt_roi_mask)
            if self.mask_rcnn.mask_loss == 'softmax_x2':
                roi_mask_loss *= 2
        elif self.mask_rcnn.mask_loss == 'sigmoid_softmax':
            roi_mask, roi_mask_bginv = roi_mask
            assert roi_mask_bginv.shape[0] == n_sample
            _, n_fg_class_x2, roi_H, roi_W = roi_mask_bginv.shape
            n_fg_class = n_fg_class_x2 // 2
            roi_mask_bginv = F.reshape(
                roi_mask_bginv, (n_sample, 2, n_fg_class, roi_H, roi_W))
            # gt_roi_mask1: 0: !vis, 1, vis
            gt_roi_mask1 = - self.xp.ones_like(gt_roi_mask)
            gt_roi_mask1[gt_roi_mask == 0] = 0  # bg
            gt_roi_mask1[gt_roi_mask == 1] = 1  # vis
            gt_roi_mask1[gt_roi_mask == 2] = 0  # inv
            roi_mask_loss = F.sigmoid_cross_entropy(
                roi_mask[np.arange(n_sample), gt_roi_label - 1, :, :],
                gt_roi_mask1)
            # gt_roi_mask2: 0: bg, 1: inv
            gt_roi_mask2 = - self.xp.ones_like(gt_roi_mask)
            gt_roi_mask2[gt_roi_mask == 0] = 0  # bg
            gt_roi_mask2[gt_roi_mask == 1] = -1  # vis
            gt_roi_mask2[gt_roi_mask == 2] = 1  # inv
            roi_mask_loss += F.softmax_cross_entropy(
                roi_mask_bginv[np.arange(n_sample), :, gt_roi_label - 1, :, :],
                gt_roi_mask2)
        elif self.mask_rcnn.mask_loss in ['sigmoid_sigmoid',
                                          'sigmoid_sigmoid+']:
            roi_mask, roi_mask_visinv = roi_mask
            # gt_roi_mask1: 0: !vis, 1: vis
            # FIXME: should be - np.ones_like.
            gt_roi_mask1 = - self.xp.ones_like(gt_roi_mask)
            gt_roi_mask1[gt_roi_mask == 0] = 0  # bg
            gt_roi_mask1[gt_roi_mask == 1] = 1  # vis
            gt_roi_mask1[gt_roi_mask == 2] = 0  # inv
            roi_mask_loss = F.sigmoid_cross_entropy(
                roi_mask[np.arange(n_sample), gt_roi_label - 1, :, :],
                gt_roi_mask1)
            # gt_roi_mask2: 0: bg, 1: vis + inv
            gt_roi_mask2 = - self.xp.ones_like(gt_roi_mask)
            gt_roi_mask2[gt_roi_mask == 0] = 0  # bg
            gt_roi_mask2[gt_roi_mask == 1] = -1  # vis
            gt_roi_mask2[gt_roi_mask == 2] = 1  # inv
            roi_mask_loss += F.sigmoid_cross_entropy(
                roi_mask_visinv[np.arange(n_sample), gt_roi_label - 1, :, :],
                gt_roi_mask2)
        elif self.mask_rcnn.mask_loss in [
            'softmax_relook_softmax',
            'softmax_relook_softmax+',
            'softmax_relook_softmax+_res',
            'softmax_relook_softmax_cls',
            'softmax_relook_softmax+_cls',
            'softmax_relook_softmax_bbox',
            'softmax_relook_softmax+_bbox',
        ]:
            roi_mask1, roi_mask2 = roi_mask
            n_positive = int((gt_roi_label > 0).sum())
            roi_mask_loss = F.softmax_cross_entropy(
                roi_mask1, gt_roi_mask[:n_positive])
            if '+' in self.mask_rcnn.mask_loss:
                roi_mask2 = roi_mask1 + roi_mask2
            roi_mask_loss += F.softmax_cross_entropy(
                roi_mask2, gt_roi_mask[:n_positive])
        elif self.mask_rcnn.mask_loss in [
            'softmax_relook_softmax_tt',
            'softmax_relook_softmax+_tt',
            'softmax_relook_softmax+_tt2',
            'softmax_relook_softmax_cls_tt',
            'softmax_relook_softmax+_cls_tt',
        ]:
            (roi_mask1_1, roi_mask1_2), (roi_mask2_1, roi_mask2_2) = roi_mask
            n_positive1 = int((gt_roi_label > 0).sum())
            roi_mask_loss1 = F.softmax_cross_entropy(
                roi_mask1_1, gt_roi_mask[:n_positive1])
            if self.mask_rcnn.mask_loss == 'softmax_relook_softmax+_tt2':
                roi_mask1_2 = roi_mask1_1 + roi_mask1_2
                roi_mask_loss1_2 = F.softmax_cross_entropy(
                    roi_mask1_2, gt_roi_mask[:n_positive1])
                roi_mask_loss1 += roi_mask_loss1_2

            n_positive2 = int((gt_roi_label2 > 0).sum())
            roi_mask_loss2_1 = F.softmax_cross_entropy(
                roi_mask2_1, gt_roi_mask2[:n_positive2])
            if self.mask_rcnn.mask_loss in [
                'softmax_relook_softmax+_tt',
                'softmax_relook_softmax+_tt2',
                'softmax_relook_softmax+_cls_tt',
            ]:
                roi_mask2_2 = roi_mask2_1 + roi_mask2_2
            roi_mask_loss2_2 = F.softmax_cross_entropy(
                roi_mask2_2, gt_roi_mask2[:n_positive2])
            roi_mask_loss = \
                roi_mask_loss1 + roi_mask_loss2_1 + roi_mask_loss2_2
            # roi_mask_loss2 = (roi_mask_loss2_1 + roi_mask_loss2_2) / 2.
            # roi_mask_loss = (roi_mask_loss1 + roi_mask_loss2) / 2.
        else:
            raise ValueError

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
