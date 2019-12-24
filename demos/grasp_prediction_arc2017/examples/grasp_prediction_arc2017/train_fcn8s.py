#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import subprocess

import chainer
from chainer import cuda
import fcn

import mvtk
from mvtk.contrib.grasp_prediction_arc2017 import datasets
from mvtk.contrib.grasp_prediction_arc2017 import models
from mvtk.contrib.grasp_prediction_arc2017 import trainers


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('-d', '--disable-cv-threads',
                        help='To avoid being stuck, disable threading in cv2',
                        action='store_true')
    parser.add_argument('-p', '--project',
                        choices=['wada_icra2018',
                                 'hasegawa_iros2018',
                                 'hasegawa_master_thesis'],
                        help='project name')
    args = parser.parse_args()

    config = 0
    gpu = args.gpu

    if args.disable_cv_threads:
        import cv2
        cv2.setNumThreads(0)

    if args.project == 'wada_icra2018':
        item_data_dir = datasets.item_data.pick_re_experiment()
        bg_from_dataset_ratio = 0.7
    elif args.project == 'hasegawa_iros2018':
        item_data_dir = osp.expanduser('~/data/iros2018/datasets/ItemDataBooks6')  # NOQA
        bg_from_dataset_ratio = 0
    elif args.project == 'hasegawa_master_thesis':
        item_data_dir = osp.expanduser('~/data/master_thesis/datasets/ItemDataBooks8')  # NOQA
        bg_from_dataset_ratio = 0
    else:
        raise ValueError

    # 0. config

    vcs_version = subprocess.check_output(
        'git log -n1 --format="%h"', shell=True).strip()
    out = 'fcn8sAtOnce'
    out += '_CFG-%03d' % config
    out += '_VCS-%s' % vcs_version
    out += '_TIME-%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = osp.join(here, 'logs', out)
    os.makedirs(out)

    # 1. dataset

    dataset_train = datasets.ItemDataDataset(
        split='train',
        item_data_dir=item_data_dir,
        bg_from_dataset_ratio=bg_from_dataset_ratio,
        project=args.project,
    )
    dataset_valid = datasets.ItemDataDataset(
        split='valid',
        item_data_dir=item_data_dir,
        bg_from_dataset_ratio=bg_from_dataset_ratio,
        project=args.project,
    )

    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.class_names)

    model = models.FCN8sAtOnce(n_class=n_class)

    # pretrained_model -> model
    pretrained_model_file = osp.expanduser('~/data/arc2017/models/fcn32s_cfg012_arc2017_iter00140000_20170729.npz')  # NOQA
    mvtk.data.download(
        url='https://drive.google.com/uc?id=1xkFzn43ZQtw3f-GdB_aDrYMp1mbzQ5B3',
        path=pretrained_model_file,  # NOQA
        md5='2cd21b3b542008d08aee5403a04569bf',
    )
    pretrained_model = fcn.models.FCN32s(n_class=41)
    chainer.serializers.load_npz(pretrained_model_file, pretrained_model)
    for l1 in pretrained_model.children():
        if l1.name.startswith('up'):
            continue
        l2 = getattr(model, l1.name)
        if l1.name == 'score_fr':
            for cls_id_from, cls_id_to in \
                    dataset_train.class_id_map.items():
                if cls_id_from in [-1, 0, 41]:
                    continue
                l2.W.data[cls_id_to, :, :, :] = \
                    l1.W.data[cls_id_from, :, :, :]
                if l2.b is not None:
                    l2.b.data[cls_id_to] = l1.b.data[cls_id_from]
        else:
            assert l2.W.data.shape == l1.W.data.shape
            l2.W.data[...] = l1.W.data[...]
            if l2.b is not None:
                assert l2.b.data.shape == l1.b.data.shape
                l2.b.data[...] = l1.b.data[...]

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.Adam(alpha=1.0e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    model.upscore8_cls.disable_update()
    model.upscore8_suc.disable_update()

    # training loop

    trainer = trainers.FCNTrainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_valid=iter_valid,
        out=out,
        max_iter=60000,
        interval_validate=4000,
        interval_save=None,
    )
    trainer.train()


if __name__ == '__main__':
    main()
