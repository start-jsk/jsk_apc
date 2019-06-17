import copy
import os
import os.path as osp
import shutil

import chainer
from chainer import training
import skimage.io

import instance_occlsegm_lib


class SemanticSegmentationVisReport(training.Extension):

    def __init__(self, iterator, transform, class_names,
                 converter=chainer.dataset.concat_examples,
                 device=None, shape=(3, 3)):
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

        target = trainer.updater.get_optimizer('main').target
        vizs = []
        for batch in it:
            img, lbl_true = list(zip(*batch))
            batch = list(map(self._transform, batch))
            x = trainer.updater.converter(batch, self.device)[0]
            with chainer.using_config('enable_backprop', False), \
                    chainer.using_config('train', False):
                score = target.predictor(x)
                lbl_pred = chainer.functions.argmax(score, axis=1)
            lbl_pred = chainer.cuda.to_cpu(lbl_pred.array)

            batch_size = len(batch)
            for i in range(batch_size):
                im = img[i]
                lt = lbl_true[i]
                lp = lbl_pred[i]
                lp[lt == -1] = -1
                lt = instance_occlsegm_lib.image.label2rgb(
                    lt, im, label_names=self._class_names)
                lp = instance_occlsegm_lib.image.label2rgb(
                    lp, im, label_names=self._class_names)
                viz = instance_occlsegm_lib.image.tile([im, lt, lp])
                vizs.append(viz)
                if len(vizs) >= (self._shape[0] * self._shape[1]):
                    break
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        viz = instance_occlsegm_lib.image.tile(vizs, shape=self._shape)
        out_file = osp.join(trainer.out, 'visualizations',
                            '%08d.jpg' % trainer.updater.iteration)
        skimage.io.imsave(out_file, viz)
        out_latest_file = osp.join(trainer.out, 'visualizations/latest.jpg')
        shutil.copy(out_file, out_latest_file)
