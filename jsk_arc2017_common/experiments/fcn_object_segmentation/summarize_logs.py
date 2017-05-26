#!/usr/bin/env python

import os
import os.path as osp

import pandas as pd
import tabulate


here = osp.dirname(osp.abspath(__file__))
logs_dir = osp.join(here, 'logs')

headers = ['setting', 'name', 'epoch', 'iteration', 'valid/mean_iu']
rows = []
for setting in os.listdir(logs_dir):
    setting_dir = osp.join(logs_dir, setting)
    for log in os.listdir(setting_dir):
        log_dir = osp.join(setting_dir, log)
        if not osp.isdir(log_dir):
            continue
        try:
            log_file = osp.join(log_dir, 'log.csv')
            df = pd.read_csv(log_file)
            columns = [c for c in df.columns if not c.startswith('train')]
            df = df[columns]
            df = df.set_index(['epoch', 'iteration'])
            index_best = df['valid/mean_iu'].idxmax()
            row_best = df.ix[index_best].dropna()
        except Exception:
            continue
        rows.append([
            setting,
            log,
            row_best.index[0][0],
            row_best.index[0][1],
            100 * row_best['valid/mean_iu'].values[0],
        ])
rows.sort(key=lambda x: x[3], reverse=True)
print(tabulate.tabulate(rows, headers=headers))
