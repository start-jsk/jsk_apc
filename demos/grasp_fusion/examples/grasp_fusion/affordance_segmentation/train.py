#!/usr/bin/env python

import argparse
import datetime
import functools
import os
import os.path as osp
import socket

os.environ['MPLBACKEND'] = 'agg'  # NOQA

import chainer
from chainer.training import extensions
import fcn

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion


here = osp.dirname(osp.abspath(__file__))


def transform(in_data, model, train):
    img, depth, lbl = in_data

    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    if lbl.ndim == 2:
        lbl = lbl[:, :, None]  # HW -> HW1
    lbl = lbl.transpose(2, 0, 1)

    if train:
        imgs, depths = model.prepare([img], [depth])
        img = imgs[0]
        depth = depths[0]

    C, H, W = img.shape
    assert C == 3
    assert lbl.shape == (lbl.shape[0], H, W)

    return img, depth, lbl


def get_model_and_data(
    affordance,
    batch_size=1,
    comm=None,
    modal='rgb',
    augmentation=True,
    resolution=30,
):
    if affordance == 'suction':
        dataset_train = grasp_fusion.datasets.SuctionDataset(
            'train', augmentation=augmentation,
        )
        dataset_test = grasp_fusion.datasets.SuctionDataset('test')
    else:
        assert affordance == 'pinch'
        dataset_train = grasp_fusion.datasets.PinchDataset(
            'train', augmentation=augmentation, resolution=resolution,
        )
        dataset_test = grasp_fusion.datasets.PinchDataset(
            'test', resolution=resolution,
        )
    channel_names = dataset_train.channel_names
    out_channels = len(channel_names)

    predictor = grasp_fusion.models.FCN8sVGG16Sigmoid(
        out_channels=out_channels, modal=modal,
    )
    model = grasp_fusion.models.FCNSigmoidTrainChain(predictor)

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
            lambda x: transform(x, model=predictor, train=True),
        ),
        batch_size=batch_size,
    )
    iter_test = chainer.iterators.SerialIterator(
        chainer.datasets.TransformDataset(
            dataset_test,
            lambda x: transform(x, model=predictor, train=False),
        ),
        batch_size=1,
        repeat=False,
        shuffle=False,
    )

    return model, iter_train, iter_test, channel_names


def get_trainer(
    optimizer,
    iter_train,
    iter_test,
    channel_names,
    args,
    comm=None,
    device=None,
):
    if device is None:
        device = args.gpu

    model = optimizer.target

    converter = functools.partial(
        chainer.dataset.concat_examples, padding=(0, 0, -1),
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
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time',
         'main/loss', 'validation/main/miou']))

    trainer.extend(
        grasp_fusion.extensions.SigmoidSegmentationVisReport(
            iter_test,
            model.predictor,
            channel_names=channel_names,
            shape=(4, 2),
        ),
        trigger=(args.interval_eval, 'epoch'),
    )

    trainer.extend(
        grasp_fusion.extensions.SigmoidSegmentationEvaluator(
            iter_test,
            model.predictor,
        ),
        trigger=(args.interval_eval, 'epoch'),
    )

    trainer.extend(extensions.snapshot_object(
        target=model.predictor, filename='model_best.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/miou',
            trigger=(args.interval_eval, 'epoch')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss'], x_key='iteration',
        file_name='loss.png', trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['validation/main/miou'], x_key='iteration',
        file_name='miou.png', trigger=(args.interval_print, 'iteration')))

    return trainer


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'affordance', choices=['suction', 'pinch'], help='affordance'
    )
    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument(
        '--multi-node', action='store_true', help='use multi node'
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
    choices = ['rgb', 'depth', 'rgb+depth']
    parser.add_argument(
        '--modal', choices=choices, default=choices[0], help='input modal'
    )
    parser.add_argument(
        '--noaug', action='store_true', help='not apply data augmentation'
    )
    parser.add_argument('--max-epoch', type=float, default=48, help='epoch')
    parser.add_argument(
        '--resolution', type=int, default=30, help='pinch rotation resolution'
    )
    parser.add_argument('--pretrained-model', help='pretrained model')
    args = parser.parse_args()

    if args.multi_node:
        import chainermn
        comm = chainermn.create_communicator('hierarchical')
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
        out = osp.join(
            here, 'logs', args.affordance, now.strftime('%Y%m%d_%H%M%S.%f')
        )
    else:
        out = None
    if comm:
        args.out = comm.bcast_obj(out)
    else:
        args.out = out
    del out

    args.hostname = socket.gethostname()
    args.git_hash = grasp_fusion_lib.utils.git_hash(__file__)

    args.batch_size = args.batch_size_per_gpu * args.n_gpu
    args.lr = args.lr_base * args.batch_size
    args.momentum = 0.99

    args.interval_print = 100
    args.interval_eval = 2

    # -------------------------------------------------------------------------

    # data
    model, iter_train, iter_test, channel_names = get_model_and_data(
        affordance=args.affordance,
        batch_size=args.batch_size_per_gpu,
        comm=comm,
        modal=args.modal,
        augmentation=not args.noaug,
        resolution=args.resolution,
    )

    if args.pretrained_model:
        chainer.serializers.load_npz(args.pretrained_model, model.predictor)

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

    model.predictor.upscore2.disable_update()
    model.predictor.upscore8.disable_update()
    model.predictor.upscore_pool4.disable_update()

    # trainer
    trainer = get_trainer(
        optimizer,
        iter_train,
        iter_test,
        channel_names,
        args=args,
        comm=comm,
        device=device,
    )
    trainer.run()


if __name__ == '__main__':
    main()
