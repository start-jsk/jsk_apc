#!/usr/bin/env python

import os.path as osp

import instance_occlsegm_lib


if __name__ == '__main__':
    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(here, 'logs')
    print('# logs_dir = %s' % logs_dir)

    keys = [
        'name',
        'last_time',
        'dataset',
        'git_hash',
        'hostname',
        'model',
        'freeze',
        'epoch',
        'iteration',
        'validation/main/miou',
    ]
    instance_occlsegm_lib.utils.summarize_logs(
        logs_dir,
        keys,
        target_key=keys[-1],
        objective='max',
    )
