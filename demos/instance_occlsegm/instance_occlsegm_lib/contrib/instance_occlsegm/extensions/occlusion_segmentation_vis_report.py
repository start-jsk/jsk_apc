import copy
import os
import os.path as osp
import shutil

import chainer
from chainer import training
import numpy as np
import skimage.io

import instance_occlsegm_lib

from ..datasets import visualize_occlusion_segmentation


class OcclusionSegmentationVisReport(training.Extension):

    def __init__(self, iterator, model, transform, class_names,
                 converter=chainer.dataset.concat_examples,
                 device=None, shape=(3, 3)):
        self.model = model
        self._iterator = iterator
        self._transform = transform
        self._class_names = class_names
        self.converter = converter
        self.device = device
        self._shape = shape

    def __call__(self, trainer):
        try:
            os.makedirs(osp.join(trainer.out, 'visualizations'))
        except OSError:
            pass

        iterator = self._iterator
        it = copy.deepcopy(iterator)

        vizs = []
        for batch in it:
            img, lbl_vis_true, lbl_occ_true = zip(*batch)
            batch = list(map(self._transform, batch))
            x = trainer.updater.converter(batch, self.device)[0]
            with chainer.using_config('enable_backprop', False), \
                    chainer.using_config('train', False):
                score, score_occ = self.model(x)
                lbl_vis = chainer.functions.argmax(score, axis=1).array
                lbl_occ = chainer.functions.sigmoid(score_occ).array > 0.5
            lbl_vis = chainer.cuda.to_cpu(lbl_vis)
            lbl_occ = chainer.cuda.to_cpu(lbl_occ)
            lbl_occ = lbl_occ.transpose(0, 2, 3, 1)

            batch_size = len(batch)
            for i in range(batch_size):
                viz_true = visualize_occlusion_segmentation(
                    img[i],
                    lbl_vis_true[i],
                    lbl_occ_true[i],
                    self._class_names,
                )
                viz_pred = visualize_occlusion_segmentation(
                    img[i],
                    lbl_vis[i],
                    lbl_occ[i],
                    self._class_names,
                )
                viz = np.hstack([viz_true, viz_pred])
                vizs.append(viz)
                if len(vizs) >= (self._shape[0] * self._shape[1]):
                    break
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        viz = instance_occlsegm_lib.image.tile(
            vizs, shape=self._shape, boundary=True)
        out_file = osp.join(
            trainer.out,
            'visualizations',
            '%08d.jpg' % trainer.updater.iteration,
        )
        skimage.io.imsave(out_file, viz)
        out_latest_file = osp.join(trainer.out, 'visualizations/latest.jpg')
        shutil.copy(out_file, out_latest_file)
