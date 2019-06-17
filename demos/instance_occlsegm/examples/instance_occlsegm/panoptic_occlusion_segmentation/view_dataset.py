#!/usr/bin/env python

import argparse

from instance_occlsegm_lib.contrib import instance_occlsegm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--split',
        choices=['train', 'test'],
        default='train',
        help='dataset split',
    )
    parser.add_argument(
        '--augmentation',
        action='store_true',
        help='do augmentation',
    )
    args = parser.parse_args()

    data = instance_occlsegm.datasets.PanopticOcclusionSegmentationDataset(
        args.split, augmentation=args.augmentation
    )
    instance_occlsegm.datasets.view_panoptic_occlusion_segmentation_dataset(
        data
    )
