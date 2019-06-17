#!/usr/bin/env python

from __future__ import division

import argparse
import datetime
import functools
import os
import os.path as osp
import random
import socket

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import cv2  # NOQA

import chainer
from chainer import training
from chainer.training import extensions
import fcn
import numpy as np

import chainer_mask_rcnn as cmr

import contrib


here = osp.dirname(osp.abspath(__file__))


class MaskRcnnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, zero_to_unlabeled=False):
        self._dataset = dataset
        self.fg_class_names = dataset.class_names[1:]
        self._zero_to_unlabeled = zero_to_unlabeled

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img, bboxes, labels, masks = self._dataset.get_example(i)
        bboxes = bboxes.astype(np.float32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # 0: background -> 0: object_0
        masks = masks.astype(np.int32, copy=False)
        if self._zero_to_unlabeled:
            # in usual instance segmentation dataset, 0 means bg + invisible
            # so we should avoid computing loss for that.
            masks[masks == 0] = -1
        return img, bboxes, labels, masks


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset',
        choices=['visible+occlusion', 'synthetic', 'occlusion'],
        help='The dataset.',
    )
    parser.add_argument('--model', '-m',
                        choices=['vgg16', 'resnet50', 'resnet101'],
                        default='resnet50', help='Base model of Mask R-CNN.')
    parser.add_argument('--pooling-func', '-pf',
                        choices=['pooling', 'align', 'resize'],
                        default='align', help='Pooling function.')
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--multi-node', '-mn', action='store_true',
                        help='use multi node')
    parser.add_argument('--mask-loss', default='softmax',
                        choices=contrib.models.MaskRCNN.mask_losses,
                        help='mask loss mode')
    default_max_epoch = (180e3 * 8) / 118287 * 3  # x3
    parser.add_argument('--max-epoch', type=float,
                        default=default_max_epoch, help='epoch')
    args = parser.parse_args()

    if args.multi_node:
        import chainermn
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank

        args.n_node = comm.inter_size
        args.n_gpu = comm.size
        chainer.cuda.get_device_from_id(device).use()
    else:
        args.n_node = 1
        args.n_gpu = 1
        chainer.cuda.get_device_from_id(args.gpu).use()
        device = args.gpu

    args.seed = 0
    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(
        here,
        'logs/train_mrcnn_lbl',
        now.strftime('%Y%m%d_%H%M%S'),
    )

    # 0.00125 * 8 = 0.01  in original
    args.batch_size = 1 * args.n_gpu
    args.lr = 0.00125 * args.batch_size
    args.weight_decay = 0.0001

    # lr / 10 at 120k iteration with
    # 160k iteration * 16 batchsize in original
    args.step_size = [(120e3 / 180e3) * args.max_epoch,
                      (160e3 / 180e3) * args.max_epoch]

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Default Config
    args.min_size = 800
    args.max_size = 1333
    args.anchor_scales = (2, 4, 8, 16, 32)

    if args.dataset == 'visible+occlusion':
        train_data1 = contrib.datasets.ARC2017RealInstancesDataset(
            'train', aug='standard')
        train_data1 = MaskRcnnDataset(train_data1, zero_to_unlabeled=True)
        train_data2 = contrib.datasets.ARC2017RealInstancesDataset(
            'test', aug='standard')
        train_data2 = MaskRcnnDataset(train_data2, zero_to_unlabeled=True)
        train_data3 = contrib.datasets.ARC2017OcclusionDataset(
            'train', do_aug=True)
        train_data3 = MaskRcnnDataset(train_data3)
        train_data = chainer.datasets.ConcatenatedDataset(
            train_data1, train_data2, train_data3,
        )
    elif args.dataset == 'synthetic':
        train_data = contrib.datasets.ARC2017SyntheticInstancesDataset(
            do_aug=True, aug_level='all')
        train_data = MaskRcnnDataset(train_data)
    elif args.dataset == 'occlusion':
        train_data = contrib.datasets.ARC2017OcclusionDataset(
            'train', do_aug=True)
        train_data = MaskRcnnDataset(train_data)
    else:
        raise ValueError
    test_data = contrib.datasets.ARC2017OcclusionDataset('test')
    instance_class_names = test_data.class_names[1:]
    test_data_list = test_data.get_video_datasets()
    del test_data
    test_data_list = [MaskRcnnDataset(td) for td in test_data_list]

    if args.pooling_func == 'align':
        pooling_func = cmr.functions.roi_align_2d
    elif args.pooling_func == 'pooling':
        pooling_func = chainer.functions.roi_pooling_2d
    elif args.pooling_func == 'resize':
        pooling_func = cmr.functions.crop_and_resize
    else:
        raise ValueError

    if args.model in ['resnet50', 'resnet101']:
        n_layers = int(args.model.lstrip('resnet'))
        mask_rcnn = contrib.models.MaskRCNNResNet(
            n_layers=n_layers,
            n_fg_class=len(instance_class_names),
            pooling_func=pooling_func,
            anchor_scales=args.anchor_scales,
            min_size=args.min_size,
            max_size=args.max_size,
            mask_loss=args.mask_loss,
        )
    else:
        raise ValueError
    model = contrib.models.MaskRCNNTrainChain(mask_rcnn)
    if args.multi_node or args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    if args.multi_node:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    for link in mask_rcnn.links():
        if isinstance(link, cmr.links.AffineChannel2D):
            link.disable_update()

    train_data = chainer.datasets.TransformDataset(
        train_data, cmr.datasets.MaskRCNNTransform(mask_rcnn))
    test_data_list = [
        chainer.datasets.TransformDataset(
            td, cmr.datasets.MaskRCNNTransform(mask_rcnn, train=False))
        for td in test_data_list]
    test_concat_data = chainer.datasets.ConcatenatedDataset(*test_data_list)
    if args.multi_node:
        # XXX: test_data is only used on device0
        if comm.rank != 0:
            train_data = None
            # test_data = None
        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
        # test_data = chainermn.scatter_dataset(test_data, comm)

    train_iter = chainer.iterators.SerialIterator(train_data, batch_size=1)
    test_iters = {
        i: chainer.iterators.SerialIterator(
            td, batch_size=1, repeat=False, shuffle=False)
        for i, td in enumerate(test_data_list)
    }
    test_concat_iter = chainer.iterators.SerialIterator(
        test_concat_data, batch_size=1, repeat=False, shuffle=False)

    converter = functools.partial(
        cmr.datasets.concat_examples,
        padding=0,
        # img, bboxes, labels, masks, scales
        indices_concat=[0, 2, 3, 4],  # img, _, labels, masks, scales
        indices_to_device=[0, 1],  # img, bbox
    )
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=device,
        converter=converter)

    trainer = training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=training.triggers.ManualScheduleTrigger(
                       args.step_size, 'epoch'))

    eval_interval = 1, 'epoch'
    log_interval = 20, 'iteration'
    plot_interval = 0.1, 'epoch'
    print_interval = 20, 'iteration'

    if not args.multi_node or comm.rank == 0:
        evaluator = contrib.extensions.InstanceSegmentationVOCEvaluator(
            test_iters, model.mask_rcnn, device=device,
            use_07_metric=False, label_names=instance_class_names)
        trainer.extend(evaluator, trigger=eval_interval)
        trainer.extend(
            extensions.snapshot_object(
                model.mask_rcnn, 'snapshot_model.npz'),
            trigger=training.triggers.MaxValueTrigger(
                'validation/main/mpq', eval_interval))
        args.git_hash = cmr.utils.git_hash()
        args.hostname = socket.gethostname()
        trainer.extend(fcn.extensions.ParamsReport(args.__dict__))
        trainer.extend(
            contrib.extensions.InstanceSegmentationVisReport(
                test_concat_iter, model.mask_rcnn,
                label_names=instance_class_names),
            trigger=eval_interval)
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'epoch', 'elapsed_time', 'lr',
             'main/loss',
             'main/roi_loc_loss',
             'main/roi_cls_loss',
             'main/roi_mask_loss',
             'main/rpn_loc_loss',
             'main/rpn_cls_loss',
             'validation/main/mpq']),
            trigger=print_interval,
        )
        trainer.extend(extensions.ProgressBar(update_interval=10))

        # plot
        assert extensions.PlotReport.available()
        trainer.extend(
            extensions.PlotReport(
                ['main/loss',
                 'main/roi_loc_loss',
                 'main/roi_cls_loss',
                 'main/roi_mask_loss',
                 'main/rpn_loc_loss',
                 'main/rpn_cls_loss'],
                file_name='loss.png', trigger=plot_interval,
            ),
            trigger=plot_interval,
        )
        trainer.extend(
            extensions.PlotReport(
                ['validation/main/map',
                 'validation/main/msq',
                 'validation/main/mdq',
                 'validation/main/mpq'],
                file_name='accuracy.png', trigger=plot_interval
            ),
            trigger=eval_interval,
        )

        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
