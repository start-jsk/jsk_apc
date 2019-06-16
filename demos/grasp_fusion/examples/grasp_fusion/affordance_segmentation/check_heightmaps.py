#!/usr/bin/env python

import os
import os.path as osp

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion


def main():
    dataset_dir = grasp_fusion.datasets.PinchDataset.root_dir

    print('Dataset dir: {}'.format(dataset_dir))

    heightmap_color_dir = osp.join(dataset_dir, 'heightmap-color')
    for filename in sorted(os.listdir(heightmap_color_dir)):
        print(filename)

        old_color_file = osp.join(dataset_dir, 'heightmap-color', filename)
        new_color_file = osp.join(dataset_dir, 'heightmap-color2', filename)
        old_depth_file = osp.join(dataset_dir, 'heightmap-depth', filename)
        new_depth_file = osp.join(dataset_dir, 'heightmap-depth2', filename)

        old_color = grasp_fusion_lib.io.imread(old_color_file)
        new_color = grasp_fusion_lib.io.imread(new_color_file)
        old_depth = grasp_fusion_lib.io.imread(old_depth_file) / 10000.
        new_depth = grasp_fusion_lib.io.imread(new_depth_file) / 10000.

        min_value = min(old_depth.min(), new_depth.min())
        max_value = max(old_depth.max(), new_depth.max())
        viz = grasp_fusion_lib.image.tile(
            [
                old_color,
                new_color,
                grasp_fusion_lib.image.colorize_depth(
                    old_depth, min_value, max_value),
                grasp_fusion_lib.image.colorize_depth(
                    new_depth, min_value, max_value),
            ],
            (2, 2),
            boundary=True,
        )
        grasp_fusion_lib.io.imshow(viz)
        if grasp_fusion_lib.io.waitkey() == ord('q'):
            break


if __name__ == '__main__':
    main()
