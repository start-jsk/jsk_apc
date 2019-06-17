#!/usr/bin/env python

import os
import os.path as osp

import chainer
import random
from selective_dualarm_stowing.datasets import DualarmDatasetV5
from selective_dualarm_stowing.trainer import get_trainer
from selective_dualarm_stowing.utils import get_APC_pt
import yaml


def train(gpu, out, config, load_model_func, classification=False):
    # default config
    if 'resume' not in config:
        config['resume'] = False
    if 'random_state' not in config:
        config['random_state'] = random.randint(0, 9999)

    # config load
    batch_size = config['batch_size']
    cross_validation = config['cross_validation']
    random_state = config['random_state']
    resize_rate = config['resize_rate']
    test_size = config['test_size']
    threshold = config['threshold']
    with_damage = config['with_damage']
    loop = int(1 / test_size)

    print('acc threshold: {}'.format(threshold))
    print('test_size: {}'.format(test_size))

    for i in range(0, loop):
        log_dir = '{0}/{1:02d}'.format(out, i)
        if not osp.exists(log_dir):
            os.mkdir(log_dir)

        print('loop: {}'.format(i))
        print('random_state: {}'.format(config['random_state']))

        # 1. dataset
        dataset_train = DualarmDatasetV5(
            'train', random_state, resize_rate, test_size,
            cross_validation, i, True, with_damage, classification)
        dataset_val = DualarmDatasetV5(
            'val', random_state, resize_rate, test_size,
            cross_validation, i, False, with_damage, classification)

        iter_train = chainer.iterators.SerialIterator(
            dataset_train, batch_size=batch_size)
        iter_val = chainer.iterators.SerialIterator(
            dataset_val, batch_size=1, repeat=False, shuffle=False)

        # 2. model
        n_failure = len(dataset_train.failure_label)
        if classification:
            n_class = len(dataset_train.class_label)
            model = load_model_func(n_failure, n_class, threshold, get_APC_pt)
        else:
            model = load_model_func(n_failure, threshold, get_APC_pt)
        if 'train_conv' not in config:
            config['train_conv'] = True

        model.train_conv = config['train_conv']
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

        # 3. optimizer
        optimizer = chainer.optimizers.AdaGrad()
        optimizer.setup(model)

        if classification:
            log_header = [
                'iteration',
                'main/loss', 'validation/main/loss',
                'main/cls/loss', 'validation/main/cls/loss',
                'main/cls/acc', 'validation/main/cls/acc',
                'main/fail/loss', 'validation/main/fail/loss',
                'main/fail/acc', 'validation/main/fail/acc',
                'elapsed_time',
            ]
        else:
            log_header = [
                'iteration',
                'main/loss', 'validation/main/loss',
                'main/acc', 'validation/main/acc',
                'elapsed_time',
            ]

        trainer = get_trainer(
            gpu, model, optimizer,
            iter_train, iter_val,
            max_iter=config['max_iter'],
            out=log_dir,
            resume=config['resume'],
            # FOR DEBUG
            # interval_eval=10
            interval_eval=config['interval_eval'],
            interval_save=config['interval_save'],
            log_header=log_header)

        # save config
        with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)

        trainer.run()
