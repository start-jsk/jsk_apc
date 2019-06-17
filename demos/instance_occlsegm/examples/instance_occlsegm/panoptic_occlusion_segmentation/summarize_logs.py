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
    parser.add_argument('--sort', '-s', nargs='+', help='sort key (ex. name)')
    parser.add_argument('--show-range', '-r', action='store_true',
                        help='show value range')
    parser.add_argument('--as-df', action='store_true', help='as df')
    args = parser.parse_args()

    print('# logs_dir = %s' % args.logs_dir)

    keys = [
        'name',
        'last_time',
        'hostname',
        'git_hash',
        'model',
        'dataset',
        'noaugmentation',
        # 'pretrained_model',
        'notrain',
        # 'lr_base',
        'pix_loss_scale',
        'epoch',
        'iteration',
        # 'validation/main/miou/vis',
        # 'validation/main/miou/occ',
        'validation/main/miou',
        'validation/main/msq/vis',
        'validation/main/msq/occ',
        'validation/main/msq',
        'validation/main/mdq',
        'validation/main/map',
        'validation/main/mpq',
    ]
    key_remap = {
        key: key[len('validation/main/'):]
        for key in keys
        if key.startswith('validation/main/')
    }
    df = instance_occlsegm_lib.utils.summarize_logs(
        args.logs_dir,
        keys,
        target_key=keys[-1],
        objective='max',
        sort=args.sort,
        show_range=args.show_range,
        as_df=args.as_df,
        key_remap=key_remap,
    )
