#!/usr/bin/env python

from __future__ import division

import argparse
import datetime
import functools
import os.path as osp
import PIL.Image
import random
import socket

import chainer
from chainer import training
from chainer.training import extensions
import chainer_mask_rcnn as cmr
from chainercv import transforms
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox
import fcn
import numpy as np

from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV1
from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV2
from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV3
from grasp_data_generator.datasets import OIDualarmGraspDatasetV1
from grasp_data_generator.datasets import OIDualarmGraspDatasetV2
from grasp_data_generator.datasets import OIRealAnnotatedDatasetV1
from grasp_data_generator.datasets import OIRealAnnotatedDatasetV2
from grasp_data_generator.extensions import ManualScheduler
from grasp_data_generator.models import OccludedGraspMaskRCNNResNet101
from grasp_data_generator.models import OccludedGraspMaskRCNNTrainChain


thisdir = osp.dirname(osp.abspath(__file__))


class Transform(object):

    def __init__(self, occluded_mask_rcnn):
        self.occluded_mask_rcnn = occluded_mask_rcnn

    def __call__(self, in_data):
        if len(in_data) == 5:
            img, ins_label, label, sg_mask, dg_mask = in_data
            rotation = None
        elif len(in_data) == 3:
            img, ins_label, label = in_data
        else:
            img, ins_label, label, sg_mask, dg_mask, rotation = in_data

        bbox = mask_to_bbox(ins_label != 0)
        _, orig_H, orig_W = img.shape
        img = self.occluded_mask_rcnn.prepare(img)
        _, H, W = img.shape
        scale = H / orig_H
        ins_label = transforms.resize(ins_label, (H, W), PIL.Image.NEAREST)
        bbox = transforms.resize_bbox(bbox, (orig_H, orig_W), (H, W))

        if len(in_data) > 3:
            sg_mask = transforms.resize(
                sg_mask.astype(np.float32), (H, W))
            dg_mask = transforms.resize(
                dg_mask.astype(np.float32), (H, W))

        if len(in_data) == 5:
            return img, ins_label, label, bbox, scale, sg_mask, dg_mask
        elif len(in_data) == 3:
            return img, ins_label, label, bbox, scale
        else:
            return img, ins_label, label, bbox, scale, \
                sg_mask, dg_mask, rotation


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', default='v2', help='Dataset version',
        choices=['v1', 'v2', 'fv1', 'fv2', 'fv3', 'ev1', 'ev2'])
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--multi-node', action='store_true',
                        help='use multi node')
    parser.add_argument('--max-epoch', type=float,
                        default=12, help='epoch')
    parser.add_argument('--seed', '-s', type=int, default=1234)
    parser.add_argument('--lr-base', type=float,
                        default=0.0025, help='lr_base')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--finetune', action='store_true',
                        help='use finetune node')
    parser.add_argument('--no-grasp', action='store_true',
                        help='use no grasp node')
    parser.add_argument('--alpha-graspable', type=float, default=0.05)
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

    now = datetime.datetime.now()
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

    if not args.multi_node or comm.rank == 0:
        out = osp.join(thisdir, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))
    else:
        out = None

    if args.multi_node:
        args.out = comm.bcast_obj(out)
    else:
        args.out = out
    del out

    # 0.00125 * 8 = 0.01  in original
    args.batch_size = 1 * args.n_gpu
    args.lr_base = args.lr_base * args.batch_size
    args.weight_decay = 0.0001

    args.step_size = [2 / 3 * args.max_epoch, 8 / 9 * args.max_epoch]

    # -------------------------------------------------------------------------
    # Dataset
    if args.dataset == 'v1':
        args.rotate_angle = None
        train_data = OIDualarmGraspDatasetV1(
            split='train', return_rotation=False)
        # test_data = OIDualarmGraspDatasetV1(split='test', imgaug=False)
    elif args.dataset == 'v2':
        args.rotate_angle = 30
        train_data = OIDualarmGraspDatasetV2(
            split='train', return_rotation=True)
        # test_data = OIDualarmGraspDatasetV2(split='test', imgaug=False)
    elif args.dataset == 'fv1':
        args.rotate_angle = 30
        train_data = FinetuningOIDualarmGraspDatasetV1(
            split='train', return_rotation=True)
        # test_data = FinetuningOIDualarmGraspDatasetV1(
        #     split='test', imgaug=False)
    elif args.dataset == 'fv2':
        args.rotate_angle = 30
        train_data = FinetuningOIDualarmGraspDatasetV2(
            split='train', return_rotation=True)
        # test_data = FinetuningOIDualarmGraspDatasetV2(
        #     split='test', imgaug=False)
    elif args.dataset == 'fv3':
        args.rotate_angle = 30
        train_data = FinetuningOIDualarmGraspDatasetV3(
            split='train', return_rotation=True)
        # test_data = FinetuningOIDualarmGraspDatasetV2(
        #     split='test', imgaug=False)
    elif args.dataset == 'ev1':
        args.rotate_angle = 30
        train_data = OIRealAnnotatedDatasetV1(split='all', imgaug=True)
    elif args.dataset == 'ev2':
        args.rotate_angle = 30
        train_data = OIRealAnnotatedDatasetV2(split='all', imgaug=True)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(args.dataset))
    label_names = train_data.label_names

    # -------------------------------------------------------------------------
    # Model + Optimizer.
    occluded_grasp_mask_rcnn = OccludedGraspMaskRCNNResNet101(
        n_fg_class=len(label_names),
        anchor_scales=args.anchor_scales,
        min_size=args.min_size,
        max_size=args.max_size,
        rpn_dim=args.rpn_dim,
        rotate_angle=args.rotate_angle)
    occluded_grasp_mask_rcnn.nms_thresh = 0.3
    occluded_grasp_mask_rcnn.score_thresh = 0.05
    if args.pretrained_model is not None:
        chainer.serializers.load_npz(
            osp.join(thisdir, args.pretrained_model), occluded_grasp_mask_rcnn)

    if args.finetune:
        assert not args.no_grasp

    if args.no_grasp:
        assert not args.finetune

    args.grasp_branch_finetune = args.finetune
    args.mask_branch_finetune = args.no_grasp

    model = OccludedGraspMaskRCNNTrainChain(
        occluded_grasp_mask_rcnn,
        grasp_branch_finetune=args.grasp_branch_finetune,
        alpha_graspable=args.alpha_graspable,
        mask_branch_finetune=args.mask_branch_finetune)

    if args.multi_node or args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(momentum=0.9)
    if args.multi_node:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    occluded_grasp_mask_rcnn.extractor.conv1.disable_update()
    occluded_grasp_mask_rcnn.extractor.bn1.disable_update()
    occluded_grasp_mask_rcnn.extractor.res2.disable_update()
    for link in occluded_grasp_mask_rcnn.links():
        if isinstance(link, cmr.links.AffineChannel2D):
            link.disable_update()

    # -------------------------------------------------------------------------
    # Transform dataset.
    train_data = chainer.datasets.TransformDataset(
        train_data, Transform(occluded_grasp_mask_rcnn))

    # -------------------------------------------------------------------------
    # Iterator.

    if args.multi_node:
        if comm.rank != 0:
            train_data = None
        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)

    # for training
    train_iter = chainer.iterators.SerialIterator(train_data, batch_size=1)
    # test_iter = chainer.iterators.SerialIterator(
    #     test_data, batch_size=1, repeat=False, shuffle=False)

    # -------------------------------------------------------------------------
    if occluded_grasp_mask_rcnn.rotate_angle is None:
        converter = functools.partial(
            cmr.datasets.concat_examples,
            padding=0,
            # img, ins_labels, labels, bboxes, sg_masks, dg_masks, scales
            indices_concat=[0, 1, 2, 4, 5, 6],
            # img, ins_labels, labels, _, sg_masks, dg_masks, scales
            indices_to_device=[0, 3],  # img, bbox
        )
    else:
        converter = functools.partial(
            cmr.datasets.concat_examples,
            padding=0,
            # img, ins_labels, labels, bboxes,
            # sg_masks, dg_masks, scales, rotations
            indices_concat=[0, 1, 2, 4, 5, 6, 7],
            # img, ins_labels, labels, _,
            # sg_masks, dg_masks, scales, rotations
            indices_to_device=[0, 3],  # img, bbox
        )
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=device,
        converter=converter)

    trainer = training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out)

    args.warm_up = not args.no_grasp

    def lr_schedule(updater):
        warm_up_duration = 500
        warm_up_rate = 1 / 3

        iteration = updater.iteration
        if args.warm_up and iteration < warm_up_duration:
            rate = warm_up_rate \
                + (1 - warm_up_rate) * iteration / warm_up_duration
        elif iteration < (args.step_size[0] * len(train_data)):
            rate = 1
        elif iteration < (args.step_size[1] * len(train_data)):
            rate = 0.1
        else:
            rate = 0.01
        return args.lr_base * rate

    trainer.extend(ManualScheduler('lr', lr_schedule))

    # eval_interval = 1, 'epoch'
    log_interval = 20, 'iteration'
    plot_interval = 0.1, 'epoch'
    print_interval = 20, 'iteration'

    if not args.multi_node or comm.rank == 0:
        # evaluator = InstanceSegmentationVOCEvaluator(
        #     test_iter, model.occluded_grasp_mask_rcnn, device=device,
        #     use_07_metric=False, label_names=label_names)
        # trainer.extend(evaluator, trigger=eval_interval)
        # trainer.extend(
        #     extensions.snapshot_object(
        #         model.occluded_grasp_mask_rcnn, 'snapshot_model.npz'),
        #     trigger=training.triggers.MaxValueTrigger(
        #         'validation/main/mpq', eval_interval))
        model_name = model.occluded_grasp_mask_rcnn.__class__.__name__
        trainer.extend(
            chainer.training.extensions.snapshot_object(
                model.occluded_grasp_mask_rcnn,
                savefun=chainer.serializers.save_npz,
                filename='%s_model_iter_{.updater.iteration}.npz'
                         % model_name),
            trigger=(1, 'epoch'))
        args.git_hash = cmr.utils.git_hash()
        args.hostname = socket.gethostname()
        trainer.extend(fcn.extensions.ParamsReport(args.__dict__))
        # trainer.extend(
        #     InstanceSegmentationVisReport(
        #         test_iter, model.occluded_grasp_mask_rcnn,
        #         label_names=label_names),
        #     trigger=eval_interval)
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'epoch', 'elapsed_time', 'lr',
             'main/loss',
             'main/rpn_loc_loss',
             'main/rpn_cls_loss',
             'main/roi_loc_loss',
             'main/roi_cls_loss',
             'main/roi_mask_loss',
             'main/roi_sg_mask_loss',
             'main/roi_dg_mask_loss',
             'validation/main/mpq']),
            trigger=print_interval,
        )
        trainer.extend(extensions.ProgressBar(update_interval=10))

        # plot
        assert extensions.PlotReport.available()
        trainer.extend(
            extensions.PlotReport(
                ['main/loss',
                 'main/rpn_loc_loss',
                 'main/rpn_cls_loss',
                 'main/roi_loc_loss',
                 'main/roi_cls_loss',
                 'main/roi_mask_loss',
                 'main/roi_sg_mask_loss',
                 'main/roi_dg_mask_loss'],
                file_name='loss.png', trigger=plot_interval,
            ),
            trigger=plot_interval,
        )
        # trainer.extend(
        #     extensions.PlotReport(
        #         ['validation/main/map',
        #          'validation/main/msq',
        #          'validation/main/mdq',
        #          'validation/main/mpq'],
        #         file_name='accuracy.png', trigger=plot_interval
        #     ),
        #     trigger=eval_interval,
        # )

        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
