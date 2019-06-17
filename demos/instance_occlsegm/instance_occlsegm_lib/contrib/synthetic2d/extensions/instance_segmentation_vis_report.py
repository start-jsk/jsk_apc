import copy
import os
import os.path as osp
import shutil

import chainer
from chainercv.utils import apply_to_iterator
import cv2
import fcn
import numpy as np
import six

import chainer_mask_rcnn as mrcnn


class InstanceSegmentationVisReport(chainer.training.extensions.Evaluator):

    def __init__(self, iterator, target, label_names,
                 file_name='visualizations/iteration=%08d.jpg',
                 shape=(3, 3), copy_latest=True):
        super(InstanceSegmentationVisReport, self).__init__(iterator, target)
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
        pred_bboxes, pred_masks, pred_labels, pred_scores = out_values[:4]
        gt_bboxes, gt_labels, gt_masks = rest_values[:3]

        # visualize
        vizs = []
        for img, gt_bbox, gt_label, gt_mask, \
            pred_bbox, pred_label, pred_mask, pred_score \
                in six.moves.zip(imgs, gt_bboxes, gt_labels, gt_masks,
                                 pred_bboxes, pred_labels, pred_masks,
                                 pred_scores):
            keep = pred_score >= 0.7
            pred_bbox = pred_bbox[keep]
            pred_label = pred_label[keep]
            pred_mask = pred_mask[keep]
            pred_score = pred_score[keep]

            # organize input
            img = img.transpose(1, 2, 0)  # CHW -> HWC

            n_fg_class = len(self.label_names)

            # bg mask
            gt_viz_bg = mrcnn.utils.draw_instance_bboxes(
                img, gt_bbox, gt_label, n_class=n_fg_class,
                masks=gt_mask == 0, captions=self.label_names[gt_label],
                bg_class=-1)
            # visible mask
            gt_viz_vis = mrcnn.utils.draw_instance_bboxes(
                img, gt_bbox, gt_label, n_class=n_fg_class,
                masks=gt_mask == 1, captions=self.label_names[gt_label],
                bg_class=-1)
            # invisible mask
            gt_viz_inv = mrcnn.utils.draw_instance_bboxes(
                img, gt_bbox, gt_label, n_class=n_fg_class,
                masks=gt_mask == 2, captions=self.label_names[gt_label],
                bg_class=-1)
            gt_viz = np.hstack([gt_viz_bg, gt_viz_vis, gt_viz_inv])

            captions = []
            for p_score, l_name in zip(pred_score,
                                       self.label_names[pred_label]):
                caption = '{:s} {:.1%}'.format(l_name, p_score)
                captions.append(caption)
            assert np.isin(np.unique(pred_mask), [-1, 0, 1, 2]).all()
            pred_viz_bg = mrcnn.utils.draw_instance_bboxes(
                img, pred_bbox, pred_label, n_class=n_fg_class,
                masks=pred_mask == 0, captions=captions, bg_class=-1)
            pred_viz_vis = mrcnn.utils.draw_instance_bboxes(
                img, pred_bbox, pred_label, n_class=n_fg_class,
                masks=pred_mask == 1, captions=captions, bg_class=-1)
            pred_viz_inv = mrcnn.utils.draw_instance_bboxes(
                img, pred_bbox, pred_label, n_class=n_fg_class,
                masks=pred_mask == 2, captions=captions, bg_class=-1)
            pred_viz = np.hstack([pred_viz_bg, pred_viz_vis, pred_viz_inv])

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
