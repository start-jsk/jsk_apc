from __future__ import division

import numpy as np
import six


def calc_grasp_segmentation_confusion(
        pred_sg_masks, pred_dg_masks, gt_sg_masks, gt_dg_masks):
    pred_sg_masks = iter(pred_sg_masks)
    pred_dg_masks = iter(pred_dg_masks)
    gt_sg_masks = iter(gt_sg_masks)
    gt_dg_masks = iter(gt_dg_masks)

    n_sg_class = 0
    n_dg_class = 0
    sg_confusion = np.zeros((n_sg_class, n_sg_class), dtype=np.int64)
    dg_confusion = np.zeros((n_dg_class, n_dg_class), dtype=np.int64)
    for pred_sg_mask, pred_dg_mask, gt_sg_mask, gt_dg_mask in six.moves.zip(
            pred_sg_masks, pred_dg_masks, gt_sg_masks, gt_dg_masks):
        pred_sg_label = np.any(pred_sg_mask, axis=0).astype(np.int32)
        gt_sg_label = np.any(gt_sg_mask, axis=0).astype(np.int32)
        if pred_sg_label.ndim != 2 or gt_sg_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_sg_label.shape != gt_sg_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_sg_label = pred_sg_label.flatten()
        gt_sg_label = gt_sg_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        sg_lb_max = np.max((pred_sg_label, gt_sg_label))
        if sg_lb_max >= n_sg_class:
            expanded_confusion = np.zeros(
                (sg_lb_max + 1, sg_lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_sg_class, 0:n_sg_class] = sg_confusion

            n_sg_class = sg_lb_max + 1
            sg_confusion = expanded_confusion

        # Count statistics from valid pixels.
        sg_mask = gt_sg_label >= 0
        sg_confusion += np.bincount(
            n_sg_class * gt_sg_label[sg_mask].astype(int) +
            pred_sg_label[sg_mask], minlength=n_sg_class**2).reshape(
                    (n_sg_class, n_sg_class))

        pred_dg_label = np.any(pred_dg_mask, axis=0).astype(np.int32)
        gt_dg_label = np.any(gt_dg_mask, axis=0).astype(np.int32)
        if pred_dg_label.ndim != 2 or gt_dg_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_dg_label.shape != gt_dg_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_dg_label = pred_dg_label.flatten()
        gt_dg_label = gt_dg_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        dg_lb_max = np.max((pred_dg_label, gt_dg_label))
        if dg_lb_max >= n_dg_class:
            expanded_confusion = np.zeros(
                (dg_lb_max + 1, dg_lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_dg_class, 0:n_dg_class] = dg_confusion

            n_dg_class = dg_lb_max + 1
            dg_confusion = expanded_confusion

        # Count statistics from valid pixels.
        dg_mask = gt_dg_label >= 0
        dg_confusion += np.bincount(
            n_dg_class * gt_dg_label[dg_mask].astype(int) +
            pred_dg_label[dg_mask], minlength=n_dg_class**2).reshape(
                    (n_dg_class, n_dg_class))

    for iter_ in (pred_sg_masks, pred_dg_masks, gt_sg_masks, gt_dg_masks):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return {'sg': sg_confusion, 'dg': dg_confusion}


def calc_semantic_segmentation_iou(confusion):
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) -
                       np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou


def calc_result(confusion):
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}


def eval_grasp_segmentation(
        pred_sg_masks, pred_dg_masks, gt_sg_masks, gt_dg_masks):
    confusions = calc_grasp_segmentation_confusion(
        pred_sg_masks, pred_dg_masks, gt_sg_masks, gt_dg_masks)
    results = {}
    results['sg'] = calc_result(confusions['sg'])
    results['dg'] = calc_result(confusions['dg'])
    return results
