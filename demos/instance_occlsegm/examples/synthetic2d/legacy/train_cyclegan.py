#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import random

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer import training
from chainer.training import extensions
import numpy as np

import chainer_cyclegan


def train(args, dataset_train, dataset_test):
    random.seed(0)
    np.random.seed(0)

    if args.multi_node:
        import chainermn
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank

        args.n_gpu = comm.size
        args.inter_size = comm.inter_size
        args.intra_size = comm.intra_size
        args.batch_size_total = args.batch_size * args.n_gpu

        chainer.cuda.get_device(device).use()
    else:
        args.batch_size_total = args.batch_size
        chainer.cuda.get_device_from_id(args.gpu).use()
        device = args.gpu

    # Model

    G_A = chainer_cyclegan.models.ResnetGenerator()
    G_B = chainer_cyclegan.models.ResnetGenerator()
    D_A = chainer_cyclegan.models.NLayerDiscriminator()
    D_B = chainer_cyclegan.models.NLayerDiscriminator()

    if args.multi_node or args.gpu >= 0:
        G_A.to_gpu()
        G_B.to_gpu()
        D_A.to_gpu()
        D_B.to_gpu()

    # Optimizer

    args.lr = 0.0002
    args.beta1 = 0.5
    args.beta2 = 0.999

    optimizer_G_A = chainer.optimizers.Adam(
        alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
    optimizer_G_B = chainer.optimizers.Adam(
        alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
    optimizer_D_A = chainer.optimizers.Adam(
        alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
    optimizer_D_B = chainer.optimizers.Adam(
        alpha=args.lr, beta1=args.beta1, beta2=args.beta2)

    if args.multi_node:
        optimizer_G_A = chainermn.create_multi_node_optimizer(
            optimizer_G_A, comm)
        optimizer_G_B = chainermn.create_multi_node_optimizer(
            optimizer_G_B, comm)
        optimizer_D_A = chainermn.create_multi_node_optimizer(
            optimizer_D_A, comm)
        optimizer_D_B = chainermn.create_multi_node_optimizer(
            optimizer_D_B, comm)

    optimizer_G_A.setup(G_A)
    optimizer_G_B.setup(G_B)
    optimizer_D_A.setup(D_A)
    optimizer_D_B.setup(D_B)

    # Dataset

    if args.multi_node:
        if comm.rank != 0:
            dataset_train = None
            dataset_test = None
        dataset_train = chainermn.scatter_dataset(
            dataset_train, comm, shuffle=True)
        dataset_test = chainermn.scatter_dataset(dataset_test, comm)

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=args.batch_size,
        n_processes=4, shared_mem=10 ** 7)
    iter_test = chainer.iterators.SerialIterator(
        dataset_test, batch_size=args.batch_size, repeat=False, shuffle=False)

    # Updater

    epoch_count = 1
    niter = 100
    niter_decay = 100

    updater = chainer_cyclegan.updaters.CycleGANUpdater(
        iterator=iter_train,
        optimizer=dict(
            G_A=optimizer_G_A,
            G_B=optimizer_G_B,
            D_A=optimizer_D_A,
            D_B=optimizer_D_B,
        ),
        device=device,
    )

    # Trainer

    out = osp.join('logs/train_cyclegan',
                   datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    trainer = training.Trainer(
        updater, (niter + niter_decay, 'epoch'), out=out)

    @training.make_extension(trigger=(1, 'epoch'))
    def tune_learning_rate(trainer):
        epoch = trainer.updater.epoch

        lr_rate = 1.0 - (max(0, epoch + 1 + epoch_count - niter) /
                         float(niter_decay + 1))

        trainer.updater.get_optimizer('G_A').alpha *= lr_rate
        trainer.updater.get_optimizer('G_B').alpha *= lr_rate
        trainer.updater.get_optimizer('D_A').alpha *= lr_rate
        trainer.updater.get_optimizer('D_B').alpha *= lr_rate

    trainer.extend(tune_learning_rate)

    if not args.multi_node or comm.rank == 0:
        trainer.extend(
            chainer_cyclegan.extensions.CycleGANEvaluator(
                iter_test, device=device))

        trainer.extend(extensions.snapshot_object(
            target=G_A, filename='G_A_{.updater.epoch:08}.npz'),
            trigger=(1, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            target=G_B, filename='G_B_{.updater.epoch:08}.npz'),
            trigger=(1, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            target=D_A, filename='D_A_{.updater.epoch:08}.npz'),
            trigger=(1, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            target=D_B, filename='D_B_{.updater.epoch:08}.npz'),
            trigger=(1, 'epoch'))

        trainer.extend(
            extensions.LogReport(trigger=(20, 'iteration')))
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'elapsed_time',
            'loss_gen_A', 'loss_gen_B',
            'loss_dis_A', 'loss_dis_B',
            'loss_cyc_A', 'loss_cyc_B',
            'loss_idt_A', 'loss_idt_B',
        ]))
        trainer.extend(contrib.extensions.ParamsReport(args.__dict__))
        trainer.extend(
            extensions.ProgressBar(update_interval=10))

        assert extensions.PlotReport.available()
        trainer.extend(extensions.PlotReport(
            y_keys=['loss_gen_A', 'loss_gen_B'],
            x_key='iteration', file_name='loss_gen.png',
            trigger=(100, 'iteration')))
        trainer.extend(extensions.PlotReport(
            y_keys=['loss_dis_A', 'loss_dis_B'],
            x_key='iteration', file_name='loss_dis.png',
            trigger=(100, 'iteration')))
        trainer.extend(extensions.PlotReport(
            y_keys=['loss_cyc_A', 'loss_cyc_B'],
            x_key='iteration', file_name='loss_cyc.png',
            trigger=(100, 'iteration')))
        trainer.extend(extensions.PlotReport(
            y_keys=['loss_idt_A', 'loss_idt_B'],
            x_key='iteration', file_name='loss_idt.png',
            trigger=(100, 'iteration')))

    trainer.run()


class UnpairedImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

    def __len__(self):
        return min(len(self.dataset_a), len(self.dataset_b))

    def get_example(self, i):
        img_a = self.dataset_a[i]
        if isinstance(img_a, tuple):
            img_a = img_a[0]

        index_b = np.random.randint(0, len(self.dataset_b))
        img_b = self.dataset_b[index_b]
        if isinstance(img_b, tuple):
            img_b = img_b[0]

        return img_a, img_b


if __name__ == '__main__':
    import contrib

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--multi-node', action='store_true',
                        help='flag to use multi node')
    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size')
    args = parser.parse_args()

    crop_size = 286

    data_train_a = contrib.datasets.ARC2017SyntheticCachedDataset(
        split='train')
    data_train_a = contrib.datasets.ClassSegRandomCropDataset(
        data_train_a, size=crop_size)
    data_train_b = contrib.datasets.ARC2017RealDataset(split='train')
    data_train_b = contrib.datasets.ClassSegRandomCropDataset(
        data_train_b, size=crop_size)
    data_train = UnpairedImageDataset(data_train_a, data_train_b)
    data_train.split = 'train'

    data_test_a = contrib.datasets.ARC2017SyntheticCachedDataset(
        split='test')
    data_test_a = contrib.datasets.ClassSegRandomCropDataset(
        data_test_a, size=crop_size)
    data_test_b = contrib.datasets.ARC2017RealDataset(split='test')
    data_test_b = contrib.datasets.ClassSegRandomCropDataset(
        data_test_b, size=crop_size)
    data_test = UnpairedImageDataset(data_test_a, data_test_b)
    data_test.split = 'test'

    # import instance_occlsegm_lib
    # def visualize_func(dataset, index):
    #     a, b = dataset[index]
    #     return np.hstack([a, b])
    # instance_occlsegm_lib.datasets.view_dataset(data_train, visualize_func)
    # instance_occlsegm_lib.datasets.view_dataset(data_test, visualize_func)

    data_train = chainer.datasets.TransformDataset(
        data_train, chainer_cyclegan.datasets.CycleGANTransform())
    data_test = chainer.datasets.TransformDataset(
        data_test, chainer_cyclegan.datasets.CycleGANTransform(train=False))
    train(args, data_train, data_test)
