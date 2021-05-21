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
import chainercv
import fcn
import numpy as np

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


here = osp.dirname(osp.abspath(__file__))


def transform_size(in_data):
    img, lbl = in_data
    H, W, C = img.shape
    assert lbl.shape == (H, W)

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
    return img, lbl


def transform(in_data, extractor='res'):
    img, lbl = transform_size(in_data)

    if extractor == 'res':
        MEAN_RGB = (123.152, 115.903, 103.063)  # Same as MaskRCNNResNet
        img = img.astype(np.float32)
        img -= MEAN_RGB
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    else:
        assert extractor == 'vgg'
        img, lbl = fcn.datasets.transform_lsvrc2012_vgg16((img, lbl))

    return img, lbl


class OcclusionSegmentationDataset(
    instance_occlsegm.datasets.OcclusionSegmentationDataset
):

    def __getitem__(self, i):
        img, lbl, _ = super(OcclusionSegmentationDataset, self).__getitem__(i)
        return img, lbl


def get_data(name, batch_size=1, comm=None, extractor='res'):
    if name == 'voc':
        dataset_train = fcn.datasets.SBDClassSeg(split='train')
        dataset_valid = fcn.datasets.VOC2011ClassSeg(split='seg11valid')
    else:
        assert name == 'occlusion'
        dataset_train = OcclusionSegmentationDataset(split='train')
        dataset_valid = OcclusionSegmentationDataset(split='test')
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
        chainer.dataset.concat_examples, padding=(0, -1),
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

    def pred_func(x):
        model(x)
        return model.score

    trainer.extend(
        fcn.extensions.SemanticSegmentationVisReport(
            pred_func,
            iter_valid_raw,
            transform=lambda x: transform(x, extractor=extractor),
            class_names=class_names,
            device=device,
            shape=(4, 2),
        ),
        trigger=(args.interval_eval, 'epoch'))

    trainer.extend(
        chainercv.extensions.SemanticSegmentationEvaluator(
            iter_valid, model, label_names=class_names),
        trigger=(args.interval_eval, 'epoch'))

    trainer.extend(extensions.snapshot_object(
        target=model, filename='model_best.npz'),
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
    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument(
        '--multi-node', action='store_true', help='use multi node'
    )
    parser.add_argument(
        '--batch-size-per-gpu', type=int, default=1, help='batch size per gpu'
    )
    parser.add_argument(
        '--lr-base', type=float, default=1e-10, help='learning rate base'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0001, help='weight decay'
    )
    parser.add_argument('--max-epoch', type=float, default=12, help='epoch')
    choices_model = ['fcn16s_resnet50', 'fcn32s_vgg16']
    parser.add_argument(
        '--model',
        choices=choices_model,
        default=choices_model[0],
        help='model',
    )
    choices_dataset = ['voc', 'occlusion']
    parser.add_argument(
        '--dataset',
        default=choices_dataset[0],
        choices=choices_dataset,
        help='dataset'
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

    args.hostname = socket.gethostname()
    args.git_hash = instance_occlsegm_lib.utils.git_hash(__file__)

    args.batch_size = args.batch_size_per_gpu * args.n_gpu
    args.lr = 1e-10 * args.batch_size
    args.momentum = 0.99

    args.interval_print = 100
    args.interval_eval = 0.5

    # -------------------------------------------------------------------------

    if 'resnet' in args.model:
        extractor = 'res'
    else:
        assert 'vgg' in args.model
        extractor = 'vgg'

    # data
    class_names, iter_train, iter_valid, iter_valid_raw = get_data(
        name=args.dataset,
        batch_size=args.batch_size_per_gpu,
        comm=comm,
        extractor=extractor,
    )
    n_class = len(class_names)

    # model
    if args.model == 'fcn16s_resnet50':
        model = instance_occlsegm.models.FCN16sResNet(n_class=n_class)
    else:
        assert args.model == 'fcn32s_vgg16'
        vgg = fcn.models.VGG16()
        chainer.serializers.load_npz(vgg.download(), vgg)
        model = fcn.models.FCN32s(n_class=n_class)
        model.init_from_vgg16(vgg)
        del vgg

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
        model.extractor.conv1.disable_update()
        model.extractor.bn1.disable_update()
        model.extractor.res2.disable_update()
        for link in model.links():
            if isinstance(link, cmr.links.AffineChannel2D):
                link.disable_update()
    else:
        model.upscore.disable_update()

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
