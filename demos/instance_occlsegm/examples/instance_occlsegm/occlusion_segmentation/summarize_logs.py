#!/usr/bin/env python

import argparse
import os.path as osp

import instance_occlsegm_lib


if __name__ == '__main__':
    here = osp.dirname(osp.abspath(__file__))
    default_logs_dir = osp.join(here, 'logs')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('logs_dir', default=default_logs_dir, nargs='?',
                        help='logs dir')
    args = parser.parse_args()

    print('# logs_dir = %s' % args.logs_dir)

    keys = [
        'name',
        'last_time',
        'dataset',
        'git_hash',
        'hostname',
        'model',
        'notrain_occlusion',
        # 'freeze',
        'lr',
        # 'weight_decay',
        'epoch',
        'iteration',
        'validation/main/miou/vis',
        'validation/main/miou/occ',
        'validation/main/miou',
    ]
    instance_occlsegm_lib.utils.summarize_logs(
        args.logs_dir,
        keys,
        target_key=keys[-1],
        objective='max',
    )
