#!/usr/bin/env python

import argparse

import grasp_fusion_lib

from grasp_fusion_lib.contrib import grasp_fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', default='train',
                        choices=['train', 'test'])
    args = parser.parse_args()

    dataset = grasp_fusion.datasets.SuctionDataset(
        args.split, augmentation=True
    )
    grasp_fusion_lib.datasets.view_dataset(
        dataset,
        grasp_fusion.datasets.SuctionDataset.visualize,
    )


if __name__ == '__main__':
    main()
