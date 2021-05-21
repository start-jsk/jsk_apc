#!/usr/bin/env python

from __future__ import division

import argparse
import datetime
import functools
import os
import os.path as osp
import random
import socket

os.environ['MPLBACKEND'] = 'agg'  # NOQA

import cv2  # NOQA

import chainer
from chainer import training
from chainer.training import extensions
import fcn
import numpy as np

import chainer_mask_rcnn as cmr

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm
from instance_occlsegm_lib.contrib import synthetic2d


here = osp.dirname(osp.abspath(__file__))


def transform_visible_only_to_with_occlusion(in_data):
    img, bboxes, labels, masks = in_data
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
    parser.add_argument('--pooling-func', '-p',
                        choices=['pooling', 'align', 'resize'],
                        default='align', help='Pooling function.')
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--multi-node', action='store_true',
                        help='use multi node')
    default_max_epoch = (180e3 * 8) / 118287 * 4  # x4
    parser.add_argument('--max-epoch', type=float,
                        default=default_max_epoch, help='epoch')
    args = parser.parse_args()

    if args.multi_node:
        import chainermn
        comm = chainermn.create_communicator('pure_nccl')
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

    if not args.multi_node or comm.rank == 0:
        out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))
    else:
        out = None
    if args.multi_node:
        args.out = comm.bcast_obj(out)
    else:
        args.out = out
    del out

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
    # args.min_size = 800
    # args.max_size = 1333
    # args.anchor_scales = (2, 4, 8, 16, 32)
    args.min_size = 600
    args.max_size = 1000
    args.anchor_scales = (4, 8, 16, 32)
    args.rpn_dim = 512

    # -------------------------------------------------------------------------
    # Dataset

    if args.dataset == 'visible+occlusion':
        train_data1 = instance_occlsegm_lib.datasets.apc.\
            ARC2017InstanceSegmentationDataset('train', aug='standard')
        train_data1 = chainer.datasets.TransformDataset(
            train_data1, transform_visible_only_to_with_occlusion)
        train_data2 = synthetic2d.datasets.ARC2017InstanceSegmentationDataset(
            'test', aug='standard')
        train_data2 = chainer.datasets.TransformDataset(
            train_data2, transform_visible_only_to_with_occlusion)
        train_data3 = synthetic2d.datasets.ARC2017OcclusionDataset(
            'train', do_aug=True)
        train_data = chainer.datasets.ConcatenatedDataset(
            train_data1, train_data2, train_data3,
        )
    elif args.dataset == 'synthetic':
        train_data = synthetic2d.datasets.ARC2017SyntheticInstancesDataset(
            do_aug=True, aug_level='all')
    elif args.dataset == 'occlusion':
        train_data = synthetic2d.datasets.ARC2017OcclusionDataset(
            'train', do_aug=True)
    else:
        raise ValueError
    test_data = synthetic2d.datasets.ARC2017OcclusionDataset('test')
    fg_class_names = test_data.class_names
    args.class_names = fg_class_names.tolist()
    test_data_list = test_data.get_video_datasets()
    del test_data

    # -------------------------------------------------------------------------
    # Model + Optimizer.

    if args.pooling_func == 'align':
        pooling_func = cmr.functions.roi_align_2d
    elif args.pooling_func == 'pooling':
        pooling_func = chainer.functions.roi_pooling_2d
    elif args.pooling_func == 'resize':
        pooling_func = cmr.functions.crop_and_resize
    else:
        raise ValueError

    args.mask_loss = 'softmax'
    assert args.model in ['resnet50', 'resnet101']
    n_layers = int(args.model.lstrip('resnet'))
    mask_rcnn = instance_occlsegm.models.MaskRCNNResNet(
        n_layers=n_layers,
        n_fg_class=len(fg_class_names),
        pooling_func=pooling_func,
        anchor_scales=args.anchor_scales,
        min_size=args.min_size,
        max_size=args.max_size,
        rpn_dim=args.rpn_dim,
    )
    mask_rcnn.nms_thresh = 0.3
    mask_rcnn.score_thresh = 0.05

    model = instance_occlsegm.models.MaskRCNNTrainChain(mask_rcnn)
    if args.multi_node or args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    if args.multi_node:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    mask_rcnn.extractor.conv1.disable_update()
    mask_rcnn.extractor.bn1.disable_update()
    mask_rcnn.extractor.res2.disable_update()
    for link in mask_rcnn.links():
        if isinstance(link, cmr.links.AffineChannel2D):
            link.disable_update()

    # -------------------------------------------------------------------------
    # Iterator.

    train_data = chainer.datasets.TransformDataset(
        train_data, instance_occlsegm.datasets.MaskRCNNTransform(mask_rcnn))
    test_data_list = [
        chainer.datasets.TransformDataset(
            td,
            instance_occlsegm.datasets.MaskRCNNTransform(
                mask_rcnn, train=False
            ),
        )
        for td in test_data_list]
    test_concat_data = chainer.datasets.ConcatenatedDataset(*test_data_list)
    if args.multi_node:
        if comm.rank != 0:
            train_data = None
        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)

    # for training
    train_iter = chainer.iterators.SerialIterator(train_data, batch_size=1)
    # for evaluation
    test_iters = {
        i: chainer.iterators.SerialIterator(
            td, batch_size=1, repeat=False, shuffle=False)
        for i, td in enumerate(test_data_list)
    }
    # for visualization
    test_concat_iter = chainer.iterators.SerialIterator(
        test_concat_data, batch_size=1, repeat=False, shuffle=False)

    # -------------------------------------------------------------------------

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
        evaluator = synthetic2d.extensions.InstanceSegmentationVOCEvaluator(
            test_iters, model.mask_rcnn, device=device,
            use_07_metric=False, label_names=fg_class_names)
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
            synthetic2d.extensions.InstanceSegmentationVisReport(
                test_concat_iter, model.mask_rcnn,
                label_names=fg_class_names),
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
