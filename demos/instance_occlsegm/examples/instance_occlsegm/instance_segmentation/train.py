#!/usr/bin/env python

import os.path as osp

import instance_occlsegm_lib

import train_common


here = osp.dirname(osp.abspath(__file__))


def main():
    args = train_common.parse_args()

    args.logs_dir = osp.join(here, 'logs')

    # Dataset.
    args.dataset = 'arc2017_real'
    train_data = instance_occlsegm_lib.datasets.apc.\
        ARC2017InstanceSegmentationDataset(split='train', aug='standard')
    test_data = instance_occlsegm_lib.datasets.apc.\
        ARC2017InstanceSegmentationDataset(split='test')
    args.class_names = tuple(test_data.class_names)

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
