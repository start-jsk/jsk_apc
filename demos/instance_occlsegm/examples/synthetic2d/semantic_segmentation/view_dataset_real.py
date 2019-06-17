#!/usr/bin/env python

import argparse

import instance_occlsegm_lib


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--split', '-s', default='train', choices=['train', 'test']
    )
    args = parser.parse_args()
    print(args)

    dataset = instance_occlsegm_lib.datasets.apc.\
        ARC2017SemanticSegmentationDataset(split=args.split)
    instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
