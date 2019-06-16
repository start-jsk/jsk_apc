#!/usr/bin/env python

import argparse

import grasp_fusion_lib

from grasp_fusion_lib.contrib.grasp_fusion.datasets import PinchDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', default='train',
                        choices=['train', 'test'])
    parser.add_argument('--resolution', type=int, default=30,
                        help='resolution of pinch rotation sampling [degree]')
    args = parser.parse_args()

    dataset = PinchDataset(
        args.split,
        augmentation=True,
        resolution=args.resolution,
    )
    grasp_fusion_lib.datasets.view_dataset(dataset, PinchDataset.visualize)


if __name__ == '__main__':
    main()
