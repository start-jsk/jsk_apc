#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import rospkg
import yaml

from selective_dualarm_stowing.models import DualarmAlex
from selective_dualarm_stowing.utils import copy_alex_chainermodel

from train import train


def load_model(n_failure, n_class, threshold, pt_func):
    rospack = rospkg.RosPack()
    model = DualarmAlex(n_failure, n_class, threshold, pt_func)
    alex_chainermodel = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'data/models/bvlc_alexnet.chainermodel')
    copy_alex_chainermodel(alex_chainermodel, model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('--cfg', type=str, default=None)
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out
    rospack = rospkg.RosPack()

    if out is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out = osp.join(
            osp.expanduser(
                '~/.ros/selective_dualarm_stowing/out/dualarm_alexnet'),
            timestamp)
        os.makedirs(out)

    # config
    if args.cfg is None:
        cfgpath = osp.join(
            rospack.get_path('selective_dualarm_stowing'),
            'cfg/dualarm_alexnet/config.yaml')
    else:
        cfgpath = args.cfg
    with open(cfgpath, 'r') as f:
        config = yaml.load(f)

    train(gpu, out, config, load_model_func=load_model,
          classification=True)


if __name__ == '__main__':
    main()
