#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import socket

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer import training
from chainer.training import extensions
import chainercv
import fcn
import numpy as np

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import synthetic2d


class Transform(object):

    def __init__(self, train=False):
        self._train = train

    def __call__(self, in_data):
        img, lbl = in_data
        if self._train:
            random_state = np.random.RandomState()
            obj_data = [{'img': img, 'lbl': lbl}]
            obj_data = instance_occlsegm_lib.aug.augment_object_data(
                obj_data, random_state)
            obj_datum = next(obj_data)
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        img = fcn.datasets.transform_lsvrc2012_vgg16((img,))[0]
        return img, lbl


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    args.max_iteration = 10000
    args.interval_eval = 1000
    args.interval_print = 10

    args.git_hash = instance_occlsegm_lib.utils.git_hash(__file__)
    args.hostname = socket.gethostname()

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))
    try:
        os.makedirs(args.out)
    except OSError:
        pass

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    data = synthetic2d.datasets.InstanceImageDataset(load_pred=False)
    class_names = data.class_names
    data_train = chainer.datasets.TransformDataset(data, Transform(train=True))
    iter_train = chainer.iterators.SerialIterator(data_train, batch_size=1)
    iter_test = chainer.iterators.SerialIterator(data, batch_size=1)

    model = synthetic2d.models.FCN8sAtOnce(n_class=len(class_names))
    vgg16 = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg16.download(), vgg16)
    model.init_from_vgg16(vgg16)
    model = chainercv.links.PixelwiseSoftmaxClassifier(predictor=model)
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=1e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    model.predictor.upscore2.disable_update()
    model.predictor.upscore_pool4.disable_update()
    model.predictor.upscore8.disable_update()

    updater = training.StandardUpdater(iter_train, optimizer, device=args.gpu)

    trainer = training.Trainer(
        updater, stop_trigger=(args.max_iteration, 'iteration'), out=args.out)

    trainer.extend(synthetic2d.extensions.ParamsReport(args.__dict__))

    trainer.extend(extensions.snapshot_object(
        target=model.predictor, filename='model_{.updater.iteration:08}.npz'),
        trigger=(args.interval_eval, 'iteration'))

    trainer.extend(extensions.LogReport(
        trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'main/loss']))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss'], x_key='iteration', file_name='loss.png',
        trigger=(args.interval_print, 'iteration')))

    trainer.extend(
        synthetic2d.extensions.SemanticSegmentationVisReport(
            iter_test, transform=Transform(train=False),
            class_names=class_names, device=args.gpu, shape=(15, 5)),
        trigger=(args.interval_print, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=5))

    trainer.run()


if __name__ == '__main__':
    main()
