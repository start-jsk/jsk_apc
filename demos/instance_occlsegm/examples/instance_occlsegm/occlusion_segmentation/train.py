#!/usr/bin/env python

import argparse
import datetime
import functools
import os
import os.path as osp
import socket

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.training import extensions
import chainer_mask_rcnn as cmr
import fcn
import numpy as np

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


here = osp.dirname(osp.abspath(__file__))


def transform_size(in_data):
    img, lbl, lbl_occ = in_data
    H, W, C = img.shape
    assert lbl.shape == (H, W)
    assert lbl_occ.shape == (H, W, lbl_occ.shape[2])

    H_dst = int(np.ceil(H / 16) * 16)
    W_dst = int(np.ceil(W / 16) * 16)

    img = np.pad(
        img,
        [(0, H_dst - H), (0, W_dst - W), (0, 0)],
        mode='constant',
        constant_values=0,
    )
    assert img.shape == (H_dst, W_dst, C)
    assert (img[H:H_dst, :, :] == 0).all()
    assert (img[:, W:W_dst, :] == 0).all()

    lbl = np.pad(
        lbl,
        [(0, H_dst - H), (0, W_dst - W)],
        mode='constant',
        constant_values=-1,
    )
    assert lbl.shape == (H_dst, W_dst)
    assert (lbl[H:H_dst, :] == -1).all()
    assert (lbl[:, W:W_dst] == -1).all()

    lbl_occ = np.pad(
        lbl_occ,
        [(0, H_dst - H), (0, W_dst - W), (0, 0)],
        mode='constant',
        constant_values=-1,
    )
    assert lbl_occ.shape == (H_dst, W_dst, lbl_occ.shape[2])
    assert (lbl_occ[H:H_dst, :, :] == -1).all()
    assert (lbl_occ[:, W:W_dst, :] == -1).all()
    return img, lbl, lbl_occ


def transform(in_data, extractor='res'):
    img, lbl, lbl_occ = transform_size(in_data)

    if extractor == 'res':
        MEAN_RGB = (123.152, 115.903, 103.063)  # Same as MaskRCNNResNet
        img = img.astype(np.float32)
        img -= MEAN_RGB
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    else:
        assert extractor == 'vgg'
        img, = fcn.datasets.transform_lsvrc2012_vgg16((img,))

    lbl_occ = lbl_occ.transpose(2, 0, 1)  # HWC -> CHW

    return img, lbl, lbl_occ


def get_data(batch_size=1, comm=None, extractor='res'):
    dataset_train = instance_occlsegm.datasets.OcclusionSegmentationDataset(
        split='train'
    )
    dataset_valid = instance_occlsegm.datasets.OcclusionSegmentationDataset(
        split='test'
    )
    # dataset_train = fcn.datasets.SBDClassSeg(split='train')
    # dataset_valid = fcn.datasets.VOC2011ClassSeg(split='seg11valid')
    class_names = dataset_train.class_names

    if comm:
        import chainermn
        if comm.rank != 0:
            dataset_train = None
        dataset_train = chainermn.scatter_dataset(
            dataset_train, comm, shuffle=True
        )

    iter_train = chainer.iterators.SerialIterator(
        chainer.datasets.TransformDataset(
            dataset_train,
            lambda x: transform(x, extractor=extractor),
        ),
        batch_size=batch_size,
    )
    iter_valid_raw = chainer.iterators.SerialIterator(
        chainer.datasets.TransformDataset(dataset_valid, transform_size),
        batch_size=1,
        repeat=False,
        shuffle=False,
    )
    iter_valid = chainer.iterators.SerialIterator(
        chainer.datasets.TransformDataset(
            dataset_valid,
            lambda x: transform(x, extractor=extractor),
        ),
        batch_size=1,
        repeat=False,
        shuffle=False,
    )

    return class_names, iter_train, iter_valid, iter_valid_raw


def get_trainer(
    optimizer,
    iter_train,
    iter_valid,
    iter_valid_raw,
    class_names,
    args,
    comm=None,
    device=None,
    extractor='res',
):
    if device is None:
        device = args.gpu

    model = optimizer.target

    converter = functools.partial(
        chainer.dataset.concat_examples, padding=(0, -1, -1),
    )
    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, converter=converter, device=device,
    )

    trainer = chainer.training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out)

    if comm and comm.rank != 0:
        return trainer

    trainer.extend(fcn.extensions.ParamsReport(args.__dict__))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.LogReport(
        trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PrintReport([
        'epoch',
        'iteration',
        'elapsed_time',
        'main/loss_vis',
        'main/loss_occ',
        'main/loss',
        'validation/main/miou/vis'
        'validation/main/miou/occ'
        'validation/main/miou',
    ]))

    trainer.extend(
        instance_occlsegm.extensions.OcclusionSegmentationVisReport(
            iter_valid_raw,
            model.predictor,
            transform=lambda x: transform(x, extractor=extractor),
            class_names=class_names,
            device=device,
            shape=(4, 2),
        ),
        trigger=(args.interval_eval, 'epoch'))

    trainer.extend(
        instance_occlsegm.extensions.OcclusionSegmentationEvaluator(
            iter_valid, model.predictor, label_names=class_names),
        trigger=(args.interval_eval, 'epoch'))

    trainer.extend(extensions.snapshot_object(
        target=model.predictor, filename='model_best.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/miou',
            trigger=(args.interval_eval, 'epoch')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss', 'main/loss_vis', 'main/loss_occ'],
        x_key='iteration',
        file_name='loss.png', trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=[
            'validation/main/miou',
            'validation/main/miou/vis',
            'validation/main/miou/occ',
        ],
        x_key='iteration',
        file_name='miou.png', trigger=(args.interval_print, 'iteration')))

    return trainer


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument(
        '--multi-node', action='store_true', help='use multi node'
    )
    parser.add_argument(
        '--notrain-occlusion', action='store_true', help='not train occlusion'
    )
    parser.add_argument(
        '--batch-size-per-gpu', type=int, default=1, help='batch size per gpu'
    )
    parser.add_argument(
        '--lr-base', type=float, default=1e-4, help='learning rate base'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0001, help='weight decay'
    )
    parser.add_argument('--max-epoch', type=float, default=36, help='epoch')
    choices_model = ['fcn16s_resnet50']
    parser.add_argument(
        '--model',
        choices=choices_model,
        default=choices_model[0],
        help='model',
    )
    args = parser.parse_args()

    if args.multi_node:
        import chainermn
        comm = chainermn.create_communicator('pure_nccl')
        device = comm.intra_rank

        args.n_node = comm.inter_size
        args.n_gpu = comm.size
        chainer.cuda.get_device_from_id(device).use()
    else:
        comm = None
        args.n_node = 1
        args.n_gpu = 1
        chainer.cuda.get_device_from_id(args.gpu).use()
        device = args.gpu

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()

    if comm is None or comm.rank == 0:
        out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))
    else:
        out = None
    if comm:
        args.out = comm.bcast_obj(out)
    else:
        args.out = out
    del out

    args.dataset = 'occlusion'
    args.hostname = socket.gethostname()
    args.git_hash = instance_occlsegm_lib.utils.git_hash(__file__)

    args.batch_size = args.batch_size_per_gpu * args.n_gpu
    args.lr = args.lr_base * args.batch_size
    args.momentum = 0.99

    args.interval_print = 20
    args.interval_eval = 0.5

    # -------------------------------------------------------------------------

    if 'resnet' in args.model:
        extractor = 'res'
    else:
        assert 'vgg' in args.model
        extractor = 'vgg'

    # data
    class_names, iter_train, iter_valid, iter_valid_raw = get_data(
        batch_size=args.batch_size_per_gpu, comm=comm, extractor=extractor,
    )
    n_class = len(class_names)

    # model
    if args.model == 'fcn16s_resnet50':
        predictor = instance_occlsegm.models.FCN16sResNetOcclusion(
            n_class=n_class
        )
        model = instance_occlsegm.models.OcclusionSegmentationTrainChain(
            predictor, train_occlusion=not args.notrain_occlusion
        )
    else:
        raise ValueError

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(
        lr=args.lr, momentum=args.momentum)
    if comm:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)

    if extractor == 'res':
        model.predictor.extractor.conv1.disable_update()
        model.predictor.extractor.bn1.disable_update()
        model.predictor.extractor.res2.disable_update()
        for link in model.links():
            if isinstance(link, cmr.links.AffineChannel2D):
                link.disable_update()
    else:
        raise ValueError
        # model.upscore.disable_update()

    # trainer
    trainer = get_trainer(
        optimizer,
        iter_train,
        iter_valid,
        iter_valid_raw,
        class_names=class_names,
        args=args,
        comm=comm,
        device=device,
        extractor=extractor,
    )
    trainer.run()


if __name__ == '__main__':
    main()
