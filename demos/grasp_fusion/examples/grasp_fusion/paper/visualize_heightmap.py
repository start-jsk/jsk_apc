#!/usr/bin/env python

import os
import os.path as osp

import numpy as np

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion


def main():
    dataset_dir = grasp_fusion.datasets.PinchDataset.root_dir

    print('Dataset dir: {}'.format(dataset_dir))

    heightmap_color_dir = osp.join(dataset_dir, 'color-input')
    for filename in sorted(os.listdir(heightmap_color_dir)):
        print(filename)
        filename2 = filename.split('-')[0] + '.png'
        print(filename2)

        if filename2 != '000002.png':
            continue

        rgb_file = osp.join(dataset_dir, 'color-input', filename)
        rgb = grasp_fusion_lib.io.imread(rgb_file)[:, :, :3]

        depth_file = osp.join(dataset_dir, 'depth-input', filename)
        depth = grasp_fusion_lib.io.imread(
            depth_file).astype(np.float32) / 10000.
        depth[depth == 0] = np.nan
        depth_viz = grasp_fusion_lib.image.colorize_depth(depth)

        heightmap_rgb_file = osp.join(
            dataset_dir, 'heightmap-color', filename2
        )
        heightmap_rgb = grasp_fusion_lib.io.imread(heightmap_rgb_file)
        heightmap_depth_file = osp.join(
            dataset_dir, 'heightmap-depth', filename2
        )
        heightmap_depth = grasp_fusion_lib.io.imread(
            heightmap_depth_file) / 10000.
        heightmap_depth_viz = grasp_fusion_lib.image.colorize_depth(
            heightmap_depth)

        grasp_fusion_lib.io.imsave('logs/heightmap/raw_rgb.jpg', rgb)
        grasp_fusion_lib.io.imsave('logs/heightmap/raw_depth.jpg', depth_viz)
        grasp_fusion_lib.io.imsave(
            'logs/heightmap/heightmap_rgb.jpg', heightmap_rgb)
        grasp_fusion_lib.io.imsave(
            'logs/heightmap/heightmap_depth.jpg', heightmap_depth_viz
        )

        # grasp_fusion_lib.io.tileimg(
        #     [rgb, depth_viz, heightmap_rgb, heightmap_depth]
        # )
        # grasp_fusion_lib.io.show()
        break


if __name__ == '__main__':
    main()
