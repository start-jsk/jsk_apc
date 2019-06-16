import copy
import os
import os.path as osp
import shutil

import chainer
from chainer import training
import numpy as np
import skimage.io

import grasp_fusion_lib


class SigmoidSegmentationVisReport(training.Extension):

    def __init__(self, iterator, model, channel_names, shape=(3, 3)):
        self.model = model
        self._iterator = iterator
        self._channel_names = channel_names
        self._shape = shape

    def __call__(self, trainer):
        try:
            os.makedirs(osp.join(trainer.out, 'visualizations'))
        except OSError:
            pass

        iterator = self._iterator

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        vizs = []
        n_viz = self._shape[0] * self._shape[1]
        for batch in it:
            imgs, depths, gt_lbls = zip(*batch)
            with chainer.using_config('enable_backprop', False), \
                    chainer.using_config('train', False):
                scores = self.model.predict(imgs, depths)
                pred_lbls = [(score > 0).astype(np.int32) for score in scores]

            batch_size = len(batch)
            out_channels = gt_lbls[0].shape[0]
            for i in range(batch_size):
                img_i = imgs[i].transpose(1, 2, 0)
                depth_i = grasp_fusion_lib.image.colorize_depth(
                    depths[i],
                    min_value=0,
                    max_value=self.model.depth_max_value,
                )

                viz = []
                for c in range(out_channels):
                    channel_name = self._channel_names[c]
                    label_names = ['not %s' % channel_name, channel_name]
                    gt_viz = grasp_fusion_lib.image.label2rgb(
                        gt_lbls[i][c],
                        img_i,
                        label_names=label_names,
                    )
                    pred_viz = grasp_fusion_lib.image.label2rgb(
                        pred_lbls[i][c],
                        img_i,
                        label_names=label_names,
                    )
                    viz_c = np.vstack([gt_viz, pred_viz])
                    viz.append(viz_c)
                viz0 = grasp_fusion_lib.image.tile(
                    viz, shape=(1, out_channels))
                viz1 = grasp_fusion_lib.image.resize(
                    img_i, height=viz0.shape[0])
                viz2 = grasp_fusion_lib.image.resize(
                    depth_i, height=viz0.shape[0])
                if self.model.modal == 'rgb':
                    viz = np.hstack((viz1, viz0))
                elif self.model.modal == 'depth':
                    viz = np.hstack((viz2, viz0))
                else:
                    assert self.model.modal == 'rgb+depth'
                    viz = np.hstack((viz1, viz2, viz0))
                vizs.append(viz)
                if len(vizs) >= n_viz:
                    break
            if len(vizs) >= n_viz:
                break

        viz = grasp_fusion_lib.image.tile(
            vizs, shape=self._shape, boundary=True)
        out_file = osp.join(
            trainer.out,
            'visualizations',
            '%08d.jpg' % trainer.updater.iteration,
        )
        skimage.io.imsave(out_file, viz)
        out_latest_file = osp.join(trainer.out, 'visualizations/latest.jpg')
        shutil.copy(out_file, out_latest_file)
