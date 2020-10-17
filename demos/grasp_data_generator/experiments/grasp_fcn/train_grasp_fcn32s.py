#!/usr/bin/env python

import argparse
import chainer
import datetime
import fcn
import grasp_data_generator
import os
import os.path as osp
import yaml


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
    outdir = osp.join(filepath, 'out')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = osp.join(outdir, timestamp)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    with open(osp.join(outdir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # datasets
    random_state = int(config['random_state'])
    dataset_train = grasp_data_generator.datasets.GraspDataset(
        'train', imgaug=True, random_state=random_state)
    dataset_valid = grasp_data_generator.datasets.GraspDataset(
        'valid', imgaug=False, random_state=random_state)

    n_class = len(dataset_train.label_names)
    assert n_class == 41

    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)

    assert config['model'] == 'grasp_fcn32s'

    model = grasp_data_generator.models.GraspFCN32s(n_class=n_class)
    vgg_path = fcn.models.VGG16.download()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg_path, vgg)
    model.init_from_vgg16(vgg)

    model.alpha = config['alpha']

    if gpu >= 0:
        model.to_gpu(gpu)

    optimizer = chainer.optimizers.Adam(alpha=config['lr'])
    optimizer.setup(model)
    optimizer.add_hook(
        chainer.optimizer.WeightDecay(rate=config['weight_decay']))
    model.upscore.disable_update()
    model.grasp_upscore.disable_update()

    trainer = grasp_data_generator.trainer.GraspTrainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_valid=iter_valid,
        out=outdir,
        max_iter=config['max_iter'],
    )

    trainer.train()

if __name__ == '__main__':
    main()
