#!/usr/bin/env python

import argparse
import os.path as osp

from summarize_logs_train_fcn import summarize_logs


if __name__ == '__main__':
    here = osp.dirname(osp.abspath(__file__))
    default_logs_dir = osp.join(here, 'logs', 'train_mrcnn_lbl')

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
        'dataset',
        'hostname',
        'git_hash',
        'model',
        # 'pretrained_model',
        # 'pooling_func',
        # 'use_pretrained',
        'mask_loss',
        'epoch',
        'iteration',
        # 'validation/main/mp_inv@0.1',
        # 'validation/main/mp_inv@0.2',
        # 'validation/main/mp_inv@0.3',
        # 'validation/main/mp_inv@0.4',
        # 'validation/main/mp_inv@0.5',
        # 'validation/main/map@50',
        # 'validation/main/map@75',
        'validation/main/map',
        # 'validation/main/map2@50',
        # 'validation/main/map2@75',
        'validation/main/msq',
        'validation/main/mdq',
        'validation/main/mpq',
        # 'validation/main/msq',
        # 'validation/main/mdq',
        # 'validation/main/msq2',
        # 'validation/main/mpq2',
    ]
    df = summarize_logs(
        args.logs_dir,
        keys,
        target_key=keys[-1],
        objective='max',
        sort=args.sort,
        show_range=args.show_range,
        as_df=args.as_df,
    )
