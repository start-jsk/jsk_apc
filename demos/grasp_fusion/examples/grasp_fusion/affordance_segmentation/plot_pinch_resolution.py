#!/usr/bin/env python

import os.path as osp

import matplotlib.pyplot as plt
import pandas

import grasp_fusion_lib


here = osp.dirname(osp.abspath(__file__))


def main():
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
    logs_dir = 'pinch'
    print('# logs_dir = %s' % logs_dir)

    logs_dir = osp.join(here, 'logs', logs_dir)
    df = grasp_fusion_lib.utils.summarize_logs(
        logs_dir,
        keys,
        target_key=keys[-1],
        objective='max',
        show_range=False,
        as_df=True,
    )
    df = df[df['noaug'] == True]  # NOQA
    df = df[df['modal'] == 'rgb+depth']
    df = df.dropna()
    df['validation/main/miou'] = pandas.to_numeric(df['validation/main/miou'])
    df = df.sort_values('resolution')
    print(df)
    df.plot('resolution', 'validation/main/miou', marker='x')
    plt.xlim(0, 45)
    plt.ylim(0, 1)
    plt.xlabel('Resolution [degree]')
    plt.ylabel('mIOU')
    plt.show()


if __name__ == '__main__':
    main()
