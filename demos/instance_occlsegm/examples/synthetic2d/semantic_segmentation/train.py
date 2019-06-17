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
from chainercv.extensions import SemanticSegmentationEvaluator
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
                obj_data, random_state, fit_output=False)
            obj_datum = next(obj_data)
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        img = fcn.datasets.transform_lsvrc2012_vgg16((img,))[0]
        return img, lbl


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    choices_dataset = ['real', 'synthetic']
    parser.add_argument('dataset', choices=choices_dataset, help='dataset')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    choices_model = ['fcn8s', 'fcn32s']
    parser.add_argument('-m', '--model', choices=choices_model,
                        default=choices_model[0], help='model file')
    parser.add_argument('--freeze', choices=['conv5', 'fc6', 'fc7'],
                        help='end layer to freeze feature extractor')
    args = parser.parse_args()

    args.max_iteration = int(100e3)
    args.interval_eval = int(1e3)
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

    if args.dataset == 'real':
        data_train = instance_occlsegm_lib.datasets.apc.\
            ARC2017SemanticSegmentationDataset(split='train')
        class_names = data_train.class_names
        data_train = chainer.datasets.TransformDataset(
            data_train, Transform(train=True))
    elif args.dataset == 'synthetic':
        data_train = synthetic2d.datasets.ARC2017SyntheticDataset(
            do_aug=True, aug_level='object')
        class_names = data_train.class_names
        data_train = chainer.datasets.TransformDataset(
            data_train, Transform(train=True))
    else:
        raise ValueError
    iter_train = chainer.iterators.MultiprocessIterator(
        data_train, batch_size=1, shared_mem=10 ** 7)

    data_test = instance_occlsegm_lib.datasets.apc.\
        ARC2017SemanticSegmentationDataset(split='test')
    data_test = chainer.datasets.TransformDataset(
        data_test, Transform(train=False))
    iter_test = chainer.iterators.SerialIterator(
        data_test, batch_size=1, repeat=False, shuffle=False)

    if args.model == 'fcn8s':
        model = synthetic2d.models.FCN8sAtOnce(n_class=len(class_names))
    elif args.model == 'fcn32s':
        model = synthetic2d.models.FCN32s(n_class=len(class_names))
    else:
        raise ValueError
    vgg16 = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg16.pretrained_model, vgg16)
    model.init_from_vgg16(vgg16)
    model = chainercv.links.PixelwiseSoftmaxClassifier(predictor=model)
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=1e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    if args.model == 'FCN8sAtOnce':
        model.predictor.upscore2.disable_update()
        model.predictor.upscore_pool4.disable_update()
        model.predictor.upscore8.disable_update()
    elif args.model == 'FCN32s':
        model.predictor.upscore.disable_update()

    if args.freeze in ['conv5', 'fc6', 'fc7']:
        model.predictor.conv5_1.disable_update()
        model.predictor.conv5_2.disable_update()
        model.predictor.conv5_3.disable_update()
    if args.freeze in ['fc6', 'fc7']:
        model.predictor.fc6.disable_update()
    if args.freeze in ['fc7']:
        model.predictor.fc7.disable_update()

    updater = training.StandardUpdater(iter_train, optimizer, device=args.gpu)

    trainer = training.Trainer(
        updater, stop_trigger=(args.max_iteration, 'iteration'), out=args.out)

    trainer.extend(
        SemanticSegmentationEvaluator(
            iter_test, model.predictor, label_names=class_names),
        trigger=(args.interval_eval, 'iteration'))

    # logging
    trainer.extend(synthetic2d.extensions.ParamsReport(args.__dict__))
    trainer.extend(extensions.LogReport(
        trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time',
         'main/loss', 'validation/main/miou']))
    trainer.extend(extensions.ProgressBar(update_interval=5))
    # plotting
    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss'], x_key='iteration',
        file_name='loss.png', trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['validation/main/miou'], x_key='iteration',
        file_name='miou.png', trigger=(args.interval_print, 'iteration')))
    # snapshotting
    trainer.extend(extensions.snapshot_object(
        target=model.predictor, filename='model_best.npz'),
        trigger=training.triggers.MaxValueTrigger(
            key='validation/main/miou',
            trigger=(args.interval_eval, 'iteration')))
    # visualizing
    data_test_raw = instance_occlsegm_lib.datasets.apc.\
        ARC2017SemanticSegmentationDataset(split='test')
    iter_test_raw = chainer.iterators.SerialIterator(
        data_test_raw, batch_size=1, repeat=False, shuffle=False)
    trainer.extend(
        synthetic2d.extensions.SemanticSegmentationVisReport(
            iter_test_raw, transform=Transform(train=False),
            class_names=class_names, device=args.gpu, shape=(6, 2)),
        trigger=(args.interval_eval, 'iteration'))

    trainer.run()


if __name__ == '__main__':
    main()
