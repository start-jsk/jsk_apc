#!/usr/bin/env python

import argparse
import os.path as osp

import grasp_fusion_lib


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--grasp', nargs='*', default=['pinch', 'suction'])
    args = parser.parse_args()
    keys = [
        'name',
        'last_time',
        # 'dataset',
        'git_hash',
        'hostname',
        'modal',
        'noaug',
        'resolution',
        # 'freeze',
        'max_epoch',
        'epoch',
        'iteration',
        'validation/main/miou',
    ]
    for logs_dir in args.grasp:
        print('# logs_dir = %s' % logs_dir)

        logs_dir = osp.join(here, 'logs', logs_dir)
        grasp_fusion_lib.utils.summarize_logs(
            logs_dir,
            keys,
            target_key=keys[-1],
            objective='max',
        )

        print()
