from __future__ import division

from collections import defaultdict
import itertools

from chainercv.evaluations import calc_detection_voc_ap
import numpy as np
import six

from chainer_mask_rcnn.utils.geometry import get_mask_overlap


def mask_iou(mask_a, mask_b):
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise ValueError

    size_a = len(mask_a)
    size_b = len(mask_b)
    iou = np.zeros((size_a, size_b), dtype=np.float64)
    for i, ma in enumerate(mask_a):
        for j, mb in enumerate(mask_b):
            ov = get_mask_overlap(ma, mb, half_if_nounion=True)
            iou[i, j] = ov
    return iou


# Original work:
# https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py  # NOQA
def calc_instseg_voc_prec_rec(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.

        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.
    """
    pred_masks = iter(pred_masks)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_masks = iter(gt_masks)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    # iou0_match = defaultdict(list)
    iou_match = defaultdict(list)   # visible + occluded
    iou1_match = defaultdict(list)  # visible
    iou2_match = defaultdict(list)  # occluded

    for pred_mask, pred_label, pred_score, gt_mask, gt_label, gt_difficult in \
            six.moves.zip(
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_mask.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_keep_l = pred_label == l
            pred_mask_l = pred_mask[pred_keep_l]
            pred_score_l = pred_score[pred_keep_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_mask_l = pred_mask_l[order]
            pred_score_l = pred_score_l[order]

            gt_keep_l = gt_label == l
            gt_mask_l = gt_mask[gt_keep_l]
            gt_difficult_l = gt_difficult[gt_keep_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_mask_l) == 0:
                continue
            if len(gt_mask_l) == 0:
                match[l].extend((0,) * pred_mask_l.shape[0])
                # iou0_match[l].extend((-1,) * pred_mask_l.shape[0])
                iou_match[l].extend((-1,) * pred_mask_l.shape[0])
                iou1_match[l].extend((-1,) * pred_mask_l.shape[0])
                iou2_match[l].extend((-1,) * pred_mask_l.shape[0])
                continue

            # iou0 = mask_iou(pred_mask_l == 0, gt_mask_l == 0)  # bg
            iou = mask_iou(
                np.isin(pred_mask_l, (1, 2)), np.isin(gt_mask_l, (1, 2))
            )   # visible + occluded
            iou1 = mask_iou(pred_mask_l == 1, gt_mask_l == 1)  # visible
            iou2 = mask_iou(pred_mask_l == 2, gt_mask_l == 2)  # occluded
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            iou_assign = iou.max(axis=1)
            # iou0_assign = iou0[np.arange(len(gt_index)), gt_index]
            iou1_assign = iou1[np.arange(len(gt_index)), gt_index]
            iou2_assign = iou2[np.arange(len(gt_index)), gt_index]
            gt_index[iou_assign < iou_thresh] = -1

            selec = np.zeros(gt_mask_l.shape[0], dtype=bool)
            for i, gt_idx in enumerate(gt_index):
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                        # iou0_match[l].append(-1)
                        iou_match[l].append(-1)
                        iou1_match[l].append(-1)
                        iou2_match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                            # iou0_match[l].append(iou0_assign[i])
                            iou_match[l].append(iou_assign[i])
                            iou1_match[l].append(iou1_assign[i])
                            iou2_match[l].append(iou2_assign[i])
                        else:
                            match[l].append(0)
                            # iou0_match[l].append(-1)
                            iou_match[l].append(-1)
                            iou1_match[l].append(-1)
                            iou2_match[l].append(-1)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
                    # iou0_match[l].append(-1)
                    iou_match[l].append(-1)
                    iou1_match[l].append(-1)
                    iou2_match[l].append(-1)

    for iter_ in (
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    sq = np.full((n_fg_class, 2), np.nan)
    dq = [np.nan] * n_fg_class
    pq = [np.nan] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        # iou0_match_l = np.array(iou0_match[l])
        iou_match_l = np.array(iou_match[l])
        iou1_match_l = np.array(iou1_match[l])
        iou2_match_l = np.array(iou2_match[l])

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        # iou0_match_l = iou0_match_l[order]
        iou_match_l = iou_match_l[order]
        iou1_match_l = iou1_match_l[order]
        iou2_match_l = iou2_match_l[order]

        # VOC Metric
        # ---------------------------------------------------------------------
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
        # ---------------------------------------------------------------------

        # PQ Metric
        # ---------------------------------------------------------------------
        if n_pos[l] <= 0:
            continue

        tp = (match_l == 1).sum()
        fp = (match_l == 0).sum()
        fn = n_pos[l] - tp
        assert tp >= 0
        assert fp >= 0
        assert fn >= 0

        keep = iou_match_l >= 0
        assert (keep == (iou1_match_l >= 0)).all()
        assert (keep == (iou2_match_l >= 0)).all()
        # iou0_match_l = iou0_match_l[keep]
        # iou_match_l = iou_match_l[keep]
        iou1_match_l = iou1_match_l[keep]
        iou2_match_l = iou2_match_l[keep]

        # SQ = \sum_{(p, q) \in TP} IoU(p, q) / TP
        # DQ = TP / (TP + 0.5 FP + 0.5 FN)
        # PQ = SQ * DQ
        # PQ = \sum_{(p, q) \in TP} IoU(p, q) / (TP + 0.5 FP + 0.5 FN)
        # if tp == 0:
        #     sq0 = 0.
        # else:
        #     sq0 = 1. * iou0_match_l.sum() / tp
        if tp == 0:
            sq1 = 0.
        else:
            sq1 = 1. * iou1_match_l.sum() / tp
        if tp == 0:
            sq2 = 0.
        else:
            sq2 = 1. * iou2_match_l.sum() / tp
        # sq[l] = (sq0 + sq1 + sq2) / 3.
        sq_l = (sq1 + sq2) / 2.
        sq[l] = (sq1, sq2)
        dq[l] = 1. * tp / (tp + (0.5 * fp) + (0.5 * fn))
        pq[l] = sq_l * dq[l]
        # ---------------------------------------------------------------------

    return prec, rec, sq, dq, pq


def eval_instseg_voc(
        pred_masks, pred_labels, pred_scores, gt_masks, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    assert not use_07_metric

    prec, rec, sq, dq, pq = calc_instseg_voc_prec_rec(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(
        prec, rec, use_07_metric=use_07_metric)

    return {
        'map': np.nanmean(ap),
        'msq': np.nanmean(np.mean(sq, axis=1)),
        'mdq': np.nanmean(dq),
        'mpq': np.nanmean(pq),
        'msq/vis': np.nanmean(sq[:, 0]),
        'msq/occ': np.nanmean(sq[:, 1]),
    }
