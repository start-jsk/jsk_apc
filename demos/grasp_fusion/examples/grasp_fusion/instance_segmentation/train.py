#!/usr/bin/env python

import os.path as osp

from grasp_fusion_lib.contrib import grasp_fusion

import train_common


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = train_common.get_parser()
    parser.add_argument(
        '--exclude-arc2017',
        action='store_true',
        help='Exclude ARC2017 objects from synthetic',
    )
    parser.add_argument(
        '--background',
        choices=['tote', 'tote+shelf'],
        default='tote',
        help='background image in 2D synthesis',
    )
    args = parser.parse_args()

    args.logs_dir = osp.join(here, 'logs')

    # Dataset.
    args.dataset = 'synthetic'
    train_data = \
        grasp_fusion.datasets.SyntheticInstanceSegmentationDataset(
            augmentation=True,
            augmentation_level='all',
            exclude_arc2017=args.exclude_arc2017,
            background=args.background,
        )
    test_data = \
        grasp_fusion.datasets.RealInstanceSegmentationDataset()
    args.class_names = tuple(test_data.class_names.tolist())

    # Model.
    args.min_size = 600
    args.max_size = 1000
    args.anchor_scales = (4, 8, 16, 32)

    # Run training!.
    train_common.train(
        args=args,
        train_data=train_data,
        test_data=test_data,
        evaluator_type='coco',
    )


if __name__ == '__main__':
    main()
