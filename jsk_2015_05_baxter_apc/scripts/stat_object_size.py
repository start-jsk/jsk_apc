#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import pickle

import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import jsk_apc2015_common


def get_object_sizes(data_dir):
    cache_file = 'object_sizes.pkl'
    if osp.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    img_shape = None
    objects = jsk_apc2015_common.get_object_list()
    df = []
    for obj in objects:
        mask_files = os.listdir(osp.join(data_dir, obj, 'masks'))
        for f in mask_files:
            if f.startswith('NP'):
                continue
            mask = cv2.imread(osp.join(data_dir, obj, 'masks', f), 0)
            if img_shape is None:
                img_shape = mask.shape
            else:
                assert img_shape == mask.shape
            mask = (mask > 127).astype(int)
            size = mask.sum()
            df.append([objects.index(obj), obj, f, size])
    df = pd.DataFrame(df)
    df.columns = ['object_index', 'object_name', 'fname', 'size']
    pickle.dump(df, open(cache_file, 'wb'))
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='APC2015berkeley dataset path')
    args = parser.parse_args()

    df = get_object_sizes(data_dir=args.data_dir)
    median_size = df.groupby('object_name').median()['size']
    df['size'] /= median_size.max()
    order = df.groupby('object_name').median().sort('size')[::-1]['object_index'].astype(np.int64)
    sns.boxplot(x='object_index', y='size', data=df, order=order)
    plt.savefig('apc2015_object_sizes.png')
