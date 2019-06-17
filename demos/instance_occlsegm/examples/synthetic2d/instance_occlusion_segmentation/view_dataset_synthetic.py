#!/usr/bin/env python

import argparse

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import synthetic2d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--aug', action='store_true', help='data aug')
    args = parser.parse_args()
    print('Args:', args)

    dataset = synthetic2d.datasets.ARC2017SyntheticInstancesDataset(
        do_aug=args.aug
    )
    instance_occlsegm_lib.datasets.view_instance_seg_dataset(
        dataset, n_mask_class=3)
