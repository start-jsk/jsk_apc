#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import yaml

import chainer
import fcn

import grasp_data_generator
from grasp_data_generator.datasets import SemanticRealAnnotatedDatasetV1
from grasp_data_generator.datasets import SemanticRealAnnotatedDatasetV2


filepath = osp.dirname(osp.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--config', '-c',
                        default='./cfg/config.yaml')
    args = parser.parse_args()

    gpu = args.gpu
    cfg_path = osp.join(filepath, args.config)

    # load config
    with open(cfg_path, 'r') as f:
        config = yaml.load(f)

    # output dir
    outdir = osp.join(filepath, 'logs')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = osp.join(outdir, timestamp)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    with open(osp.join(outdir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # datasets
    random_state = int(config['random_state'])
    if 'train_as_fcn' in config and config['train_as_fcn']:
        dataset_train = SemanticRealAnnotatedDatasetV2(
            split='all', imgaug=True)
        dataset_valid = SemanticRealAnnotatedDatasetV1(
            split='all', imgaug=False)
    else:
        if 'dataset_class' not in config:
            config['dataset_class'] = 'DualarmGraspDataset'
        dataset_class = getattr(grasp_data_generator.datasets,
                                config['dataset_class'])
        dataset_train = dataset_class('train', imgaug=True,
                                      random_state=random_state)
        dataset_valid = dataset_class('valid', imgaug=False,
                                      random_state=random_state)

    n_class = len(dataset_train.label_names)
    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)

    assert config['model'] == 'dualarm_grasp_fcn32s'

    model = grasp_data_generator.models.DualarmGraspFCN32s(n_class=n_class)
    if 'pretrained_modelpath' not in config:
        vgg_path = fcn.models.VGG16.download()
        vgg = fcn.models.VGG16()
        chainer.serializers.load_npz(vgg_path, vgg)
        model.init_from_vgg16(vgg)
    else:
        print('load pretrained model from data/pretrained_model/{}'.format(
            config['pretrained_modelpath']))
        modelpath = osp.join(
            filepath, '../../data/pretrained_model',
            config['pretrained_modelpath'])
        chainer.serializers.load_npz(modelpath, model)

    if 'alpha' in config:
        alpha = config['alpha']
        if isinstance(alpha, dict):
            model.alpha_single = alpha['single']
            model.alpha_dual = alpha['dual']
        else:
            model.alpha_single = alpha
            model.alpha_dual = 1.0
    else:
        model.alpha_single = 1.0
        model.alpha_dual = 1.0

    if 'alpha_graspable' in config:
        model.alpha_graspable = config['alpha_graspable']
    else:
        model.alpha_graspable = 20.0

    if 'frequency_balancing' in config:
        frq_balancing = config['frequency_balancing']
        model.frq_balancing = frq_balancing
    else:
        model.frq_balancing = False

    if 'use_seg_loss' in config:
        model.use_seg_loss = config['use_seg_loss']
    else:
        model.use_seg_loss = True

    if 'use_grasp_loss' in config:
        model.use_grasp_loss = config['use_grasp_loss']
    else:
        model.use_grasp_loss = True

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)

    optimizer = chainer.optimizers.Adam(alpha=config['lr'])
    optimizer.setup(model)
    optimizer.add_hook(
        chainer.optimizer.WeightDecay(rate=config['weight_decay']))
    model.upscore.disable_update()
    model.single_grasp_upscore.disable_update()
    model.dual_grasp_upscore.disable_update()

    if 'train_as_fcn' in config and config['train_as_fcn']:
        model.train_as_fcn = True
        trainer = grasp_data_generator.trainer.FCNTrainer(
            device=gpu,
            model=model,
            optimizer=optimizer,
            iter_train=iter_train,
            iter_valid=iter_valid,
            out=outdir,
            max_iter=config['max_iter'],
            interval_validate=500)
    else:
        model.train_as_fcn = False
        trainer = grasp_data_generator.trainer.DualarmGraspTrainer(
            device=gpu,
            model=model,
            optimizer=optimizer,
            iter_train=iter_train,
            iter_valid=iter_valid,
            out=outdir,
            max_iter=config['max_iter'])

    trainer.train()


if __name__ == '__main__':
    main()
