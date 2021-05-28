#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import os.path as osp

filenames = [
    'vis_cls_label',
    'vis_ins_label',
    'occ_cls_label',
    'occ_ins_label',
    'single_cls_label',
    'single_ins_label',
    'dual_cls_label',
    'dual_ins_label',
]


def main(data_dir):
    objectnames = os.listdir(data_dir)
    # copy objects
    for objectname in objectnames:
        print('object: {}'.format(objectname))
        recog_dir = osp.join(data_dir, objectname, 'recognition')
        for d in os.listdir(recog_dir):
            save_dir = osp.join(recog_dir, d)
            for filename in filenames:
                labelpath = osp.join(save_dir, '{}.png'.format(filename))
                if osp.exists(labelpath):
                    npzpath = osp.join(save_dir, '{}.npz'.format(filename))
                    label = cv2.imread(labelpath, cv2.IMREAD_GRAYSCALE)
                    label[label == 0] = -1
                    np.savez_compressed(npzpath, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data-dir', default=None)
    args = parser.parse_args()

    main(args.data_dir)
