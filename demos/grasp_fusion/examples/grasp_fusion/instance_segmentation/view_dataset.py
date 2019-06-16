#!/usr/bin/env python

import argparse

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    if args.split == 'train':
        dataset = \
            grasp_fusion.datasets.SyntheticInstanceSegmentationDataset(
                augmentation=True, exclude_arc2017=True,
            )
    else:
        dataset = \
            grasp_fusion.datasets.RealInstanceSegmentationDataset()
    grasp_fusion_lib.datasets.view_instance_seg_dataset(
        dataset, n_mask_class=2)


if __name__ == '__main__':
    main()
