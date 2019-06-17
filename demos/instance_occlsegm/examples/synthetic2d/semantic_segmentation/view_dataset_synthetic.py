#!/usr/bin/env python

import argparse

import instance_occlsegm_lib

from instance_occlsegm_lib.contrib import synthetic2d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--aug', action='store_true')
    args = parser.parse_args()

    dataset = synthetic2d.datasets.ARC2017SyntheticDataset(
        do_aug=args.aug, aug_level='all'
    )
    instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
