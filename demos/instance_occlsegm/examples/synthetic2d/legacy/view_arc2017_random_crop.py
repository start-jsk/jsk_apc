#!/usr/bin/env python

import argparse

import instance_occlsegm_lib

import contrib


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', choices=['real', 'synthetic'])
    args = parser.parse_args()

    if args.dataset == 'real':
        data = contrib.datasets.ARC2017RealDataset(split='train')
    else:
        assert args.dataset == 'synthetic'
        data = contrib.datasets.ARC2017SyntheticCachedDataset(split='train')
    data = contrib.datasets.ClassSegRandomCropDataset(data, size=286)
    instance_occlsegm_lib.datasets.view_class_seg_dataset(data)


if __name__ == '__main__':
    main()
