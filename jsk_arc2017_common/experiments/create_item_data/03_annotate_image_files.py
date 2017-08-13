#!/usr/bin/env python

import glob
import json
import os
import os.path as osp
import subprocess

import skimage.io


here = osp.dirname(osp.abspath(__file__))


def main():
    item_data_dir = osp.join(here, 'item_data')

    for item_name_upper in os.listdir(item_data_dir):
        item_dir = osp.join(item_data_dir, item_name_upper)
        if not osp.isdir(item_dir):
            continue

        item_name_lower = item_name_upper.lower()

        for fname in sorted(glob.glob(osp.join(item_dir, '*.png'))):
            labelme_file = osp.splitext(fname)[0] + '.json'
            if osp.exists(labelme_file):
                continue
            cmd = 'labelme {:s} -O {:s}'.format(fname, labelme_file)
            print('+ {:s}'.format(cmd))
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
