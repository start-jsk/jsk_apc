#!/usr/bin/env python

import argparse
import numpy as np
import os
import os.path as osp
import pandas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detail', '-d', action='store_true')
    parser.add_argument('--log', '-l')
    args = parser.parse_args()

    detail = args.detail
    logdir = args.log
    valid_fail_accs = []
    valid_cls_accs = []
    for dirname in os.listdir(logdir):
        if not osp.isdir(osp.join(logdir, dirname)):
            continue
        jsonpath = osp.join(logdir, dirname, 'log.json')
        data = pandas.read_json(jsonpath)
        columns = data.columns.values
        if 'validation/main/acc' in columns:
            valid_fail_acc = data['validation/main/acc']
            valid_fail_acc = valid_fail_acc.dropna()
            valid_fail_accs.append(valid_fail_acc.values)
            continue
        if 'validation/main/fail/acc' in columns:
            valid_fail_acc = data['validation/main/fail/acc']
            valid_fail_acc = valid_fail_acc.dropna()
            valid_fail_accs.append(valid_fail_acc.values)
        if 'validation/main/cls/acc' in columns:
            valid_cls_acc = data['validation/main/cls/acc']
            valid_cls_acc = valid_cls_acc.dropna()
            valid_cls_accs.append(valid_cls_acc.values)
    valid_fail_accs = np.array(valid_fail_accs, dtype=np.float32)
    valid_cls_accs = np.array(valid_cls_accs, dtype=np.float32)
    mean_fail_accs = np.mean(valid_fail_accs, axis=0)
    mean_cls_accs = np.mean(valid_cls_accs, axis=0)
    print('failure prediction')
    for i, acc in enumerate(mean_fail_accs):
        print('    iter: {0:03d}, acc: {1}'.format((i+1)*1000, acc))
    if detail:
        print('=============================================')
        print(valid_fail_accs)
        print('=============================================')

    print('classification')
    for i, acc in enumerate(mean_cls_accs):
        print('    iter: {0:03d}, acc: {1}'.format((i+1)*1000, acc))
    if detail:
        print('=============================================')
        print(valid_cls_accs)
        print('=============================================')


if __name__ == '__main__':
    main()
