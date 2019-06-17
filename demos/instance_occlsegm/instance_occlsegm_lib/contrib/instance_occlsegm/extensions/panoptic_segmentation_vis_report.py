import copy
import os
import os.path as osp
import shutil

import chainer
from chainercv.utils import apply_to_iterator
import cv2
import fcn
import numpy as np

import instance_occlsegm_lib

from ..datasets import visualize_occlusion_segmentation


def visualize_panoptic_occlusion_segmentation(
    img, bboxes, labels, masks, lbl_vis, lbl_occ, fg_class_names
):
    if isinstance(fg_class_names, np.ndarray):
        fg_class_names = fg_class_names.tolist()

    viz_ins = instance_occlsegm_lib.datasets.visualize_instance_segmentation(
        img, bboxes, labels, masks, fg_class_names, n_mask_class=3
    )
    class_names = ['__background__'] + fg_class_names
    viz_sem = visualize_occlusion_segmentation(
        img, lbl_vis, lbl_occ, class_names
    )
    viz = instance_occlsegm_lib.image.tile([viz_ins, viz_sem])
    return viz


class PanopticSegmentationVisReport(chainer.training.extensions.Evaluator):

    def __init__(self, iterator, target, label_names,
                 file_name='visualizations/iteration=%08d.jpg',
                 shape=(3, 3), copy_latest=True):
        super(PanopticSegmentationVisReport, self).__init__(iterator, target)
        self.label_names = np.asarray(label_names)
        self.file_name = file_name
        self._shape = shape
        self._copy_latest = copy_latest

    def __call__(self, trainer):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)
        imgs, = in_values
        (
            pred_bboxes,
            pred_masks,
            pred_labels,
            pred_scores,
            pred_lbls_vis,
            pred_lbls_occ,
        ) = out_values
        gt_bboxes, gt_labels, gt_masks, gt_lbls_vis, gt_lbls_occ = rest_values

        # visualize
        vizs = []
        for img in imgs:
            gt_bbox = next(gt_bboxes)
            gt_label = next(gt_labels)
            gt_mask = next(gt_masks)
            gt_lbl_vis = next(gt_lbls_vis)
            gt_lbl_occ = next(gt_lbls_occ)

            pred_bbox = next(pred_bboxes)
            pred_mask = next(pred_masks)
            pred_label = next(pred_labels)
            pred_score = next(pred_scores)
            pred_lbl_vis = next(pred_lbls_vis)
            pred_lbl_occ = next(pred_lbls_occ)

            keep = pred_score >= 0.7
            pred_bbox = pred_bbox[keep]
            pred_label = pred_label[keep]
            pred_mask = pred_mask[keep]
            pred_score = pred_score[keep]

            # organize input: CHW -> HWC
            img = img.transpose(1, 2, 0)
            pred_lbl_occ = pred_lbl_occ.transpose(1, 2, 0)
            gt_lbl_occ = gt_lbl_occ.transpose(1, 2, 0)

            gt_viz = visualize_panoptic_occlusion_segmentation(
                img=img,
                bboxes=gt_bbox,
                labels=gt_label,
                masks=gt_mask,
                lbl_vis=gt_lbl_vis,
                lbl_occ=gt_lbl_occ,
                fg_class_names=self.label_names
            )
            pred_viz = visualize_panoptic_occlusion_segmentation(
                img=img,
                bboxes=pred_bbox,
                labels=pred_label,
                masks=pred_mask,
                lbl_vis=pred_lbl_vis,
                lbl_occ=pred_lbl_occ,
                fg_class_names=self.label_names
            )

            viz = np.vstack([gt_viz, pred_viz])
            vizs.append(viz)
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        viz = fcn.utils.get_tile_image(vizs, tile_shape=self._shape)
        file_name = osp.join(
            trainer.out, self.file_name % trainer.updater.iteration)
        try:
            os.makedirs(osp.dirname(file_name))
        except OSError:
            pass
        cv2.imwrite(file_name, viz[:, :, ::-1])

        if self._copy_latest:
            shutil.copy(file_name,
                        osp.join(osp.dirname(file_name), 'latest.jpg'))
