#!/usr/bin/env python

import datetime
import os
import os.path as osp
import sys

import click
import pytz
import torch
import torchfcn
import torchvision
import yaml

from dataset import DatasetV1


this_dir = osp.dirname(osp.realpath(__file__))


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def main(config_file):
    config = yaml.load(open(config_file))
    assert 'max_iteration' in config
    assert 'optimizer' in config
    assert 'lr' in config
    assert 'weight_decay' in config

    name = osp.splitext(osp.basename(config_file))[0]
    config['config_file'] = osp.realpath(config_file)
    config['timestamp'] = datetime.datetime.now(
        pytz.timezone('Asia/Tokyo')).isoformat()
    config['out'] = osp.join(this_dir, 'logs',
                             '%s_%s' % (name, config['timestamp']))
    if not osp.exists(config['out']):
        os.makedirs(config['out'])
    with open(osp.join(config['out'], 'params.yaml'), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    yaml.safe_dump(config, sys.stderr, default_flow_style=False)

    cuda = torch.cuda.is_available()

    seed = 1
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        DatasetV1(split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        DatasetV1(split='valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    n_class = len(DatasetV1.class_names)
    model = torchfcn.models.FCN32s(n_class=n_class, deconv=False)
    start_epoch = 0
    if config.get('resume'):
        checkpoint = torch.load(config['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        pth_file = osp.expanduser('~/data/models/torch/vgg16-00b39a1b.pth')
        vgg16 = torchvision.models.vgg16()
        vgg16.load_state_dict(torch.load(pth_file))
        torchfcn.utils.copy_params_vgg16_to_fcn32s(vgg16, model,
                                                   init_upscore=False)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = getattr(torch.optim, config['optimizer'])
    optim = optim(model.parameters(), lr=config['lr'],
                  weight_decay=config['weight_decay'])
    if config.get('resume'):
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
