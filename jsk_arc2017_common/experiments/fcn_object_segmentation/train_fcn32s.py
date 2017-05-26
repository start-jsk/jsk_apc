#!/usr/bin/env python

import os
import os.path as osp
import sys

import click
import torch
import torchfcn
import yaml

from dataset import DatasetV1


this_dir = osp.dirname(osp.realpath(__file__))


def git_hash():
    import shlex
    import subprocess
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def load_config(config_file):
    import datetime
    import pytz

    config = yaml.load(open(config_file))
    assert 'max_iteration' in config
    assert 'optimizer' in config
    assert 'lr' in config
    assert 'weight_decay' in config
    assert 'aug' in config

    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    now = now.replace(tzinfo=None)

    out = osp.splitext(osp.basename(config_file))[0]
    setting = osp.basename(osp.dirname(osp.abspath(config_file)))
    for key, value in sorted(config.items()):
        if isinstance(value, basestring):
            value = value.replace('/', 'SLASH')
            value = value.replace(':', 'COLON')
        out += '_{key}-{value}'.format(key=key.upper(), value=value)
    out += '_VCS-%s' % git_hash()
    out += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    config['out'] = osp.join(this_dir, 'logs', setting, out)

    config['config_file'] = osp.realpath(config_file)
    config['timestamp'] = datetime.datetime.now(
        pytz.timezone('Asia/Tokyo')).isoformat()
    if not osp.exists(config['out']):
        os.makedirs(config['out'])
    with open(osp.join(config['out'], 'params.yaml'), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    return config


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--resume', type=click.Path(exists=True))
def main(config_file, resume):
    config = load_config(config_file)
    yaml.safe_dump(config, sys.stderr, default_flow_style=False)

    cuda = torch.cuda.is_available()

    seed = 1
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        DatasetV1(split='train', transform=True, aug=config['aug']),
        batch_size=config.get('batch_size', 1), shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        DatasetV1(split='valid', transform=True, aug=config['aug']),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    n_class = len(DatasetV1.class_names)
    model = torchfcn.models.FCN32s(n_class=n_class, nodeconv=True)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=False)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = getattr(torch.optim, config['optimizer'])
    optim = optim(model.parameters(), lr=config['lr'],
                  weight_decay=config['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=valid_loader,
        out=config['out'],
        max_iter=config['max_iteration'],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_epoch * len(train_loader)
    trainer.train()


if __name__ == '__main__':
    main()
