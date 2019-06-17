#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import skimage.io

import instance_occlsegm_lib


def main():
    root_dir = osp.expanduser('~/.ros/instance_occlsegm')
    for save_dir in sorted(os.listdir(root_dir)):
        save_dir = osp.join(root_dir, save_dir)
        print('-' * 79)
        print(save_dir)
        for frame_dir in sorted(os.listdir(save_dir)):
            frame_dir = osp.join(save_dir, frame_dir)
            print(frame_dir)

            img_file = osp.join(frame_dir, 'image.jpg')
            img = skimage.io.imread(img_file)
            depth_file = osp.join(frame_dir, 'depth.npz')
            depth = np.load(depth_file)['arr_0']
            depth_viz = instance_occlsegm_lib.image.colorize_depth(
                depth, min_value=0.4, max_value=0.9
            )
            # depth_viz_file = osp.join(frame_dir, 'depth_viz.jpg')
            # depth_viz = skimage.io.imread(depth_viz_file)

            viz = instance_occlsegm_lib.image.tile([img, depth_viz])
            instance_occlsegm_lib.io.imshow(viz)
            if instance_occlsegm_lib.io.waitkey() == ord('q'):
                return


if __name__ == '__main__':
    main()
