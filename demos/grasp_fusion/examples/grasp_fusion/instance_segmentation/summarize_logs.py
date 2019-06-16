#!/usr/bin/env python

import argparse
import os.path as osp

import grasp_fusion_lib


if __name__ == '__main__':
    here = osp.dirname(osp.abspath(__file__))
    default_logs_dir = osp.join(here, 'logs')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('logs_dir', default=default_logs_dir, nargs='?',
                        help='logs dir')
    parser.add_argument('--sort', '-s', nargs='+', help='sort key (ex. name)')
    # parser.add_argument('--show-range', '-r', action='store_true',
    #                     help='show value range')
    # parser.add_argument('--as-df', action='store_true', help='as df')
    args = parser.parse_args()

    print('# logs_dir = %s' % args.logs_dir)

    keys = [
        'name',
        'last_time',
        'dataset',
        'exclude_arc2017',
        'background',
        'hostname',
        'git_hash',
        'model',
        'epoch',
        'iteration',
        'validation/main/map@0.5',
        'validation/main/map@0.75',
        'validation/main/map',
    ]
    df = grasp_fusion_lib.utils.summarize_logs(
        args.logs_dir,
        keys,
        target_key=keys[-1],
        objective='max',
        sort=args.sort,
        # show_range=args.show_range,
        # as_df=args.as_df,
    )
