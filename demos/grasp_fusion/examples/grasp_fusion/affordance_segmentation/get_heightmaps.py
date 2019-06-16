#!/usr/bin/env python

# This program is re-implementation of getHeightmaps.m

from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import warnings

import numpy as np
import skimage.io
import tqdm

from grasp_fusion_lib.contrib import grasp_fusion


def get_heightmaps(grasp_type='suction'):
    print('Grasp type: {}'.format(grasp_type))

    pinch_dir = grasp_fusion.datasets.PinchDataset('train').root_dir
    suction_dir = grasp_fusion.datasets.SuctionDataset('train').root_dir

    if grasp_type == 'pinch':
        dataset_dir = pinch_dir
    else:
        assert grasp_type == 'suction'
        dataset_dir = suction_dir

    # List colot images from dataset
    color_dir = osp.join(dataset_dir, 'color-input')
    if grasp_type == 'pinch':
        color_files = sorted(glob.glob(osp.join(color_dir, '*-0.png')))
    else:
        color_files = sorted(glob.glob(osp.join(color_dir, '*.png')))

    # Create directory to save height_maps
    heightmap_color_dir = osp.join(dataset_dir, 'heightmap-color2')
    heightmap_depth_dir = osp.join(dataset_dir, 'heightmap-depth2')
    try:
        os.mkdir(heightmap_color_dir)
        os.mkdir(heightmap_depth_dir)
    except OSError:
        pass
    print('Created: {}'.format(heightmap_color_dir))
    print('Created: {}'.format(heightmap_depth_dir))

    if grasp_type == 'suction':
        heightmap_suction_dir = osp.join(dataset_dir, 'heightmap-suction2')
        try:
            os.mkdir(heightmap_suction_dir)
        except OSError:
            pass
        print('Created: {}'.format(heightmap_suction_dir))

    # Read fixed position of bin w.r.t. world coordinates
    bin_position_file = osp.join(pinch_dir, 'bin-position.txt')
    bin_middle_bottom = np.loadtxt(bin_position_file)
    # Assumes fixed orientation
    bin_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Offset w.r.t. bin coordinates
    grid_origin = bin_middle_bottom - np.dot(bin_rot, np.array([0.3, 0.2, 0]))

    # Loop through all sample and generate height maps
    for sample_idx in tqdm.tqdm(range(len(color_files))):
        sample_name = color_files[sample_idx][:-6].split('/')[-1]

        heightmaps = []
        missing_heightmaps = []
        heightmaps_color = []
        heightmaps_suction = []

        # Use every two RGB-D images (captured from two different cameras) to
        # construct a unified height map
        for cam_idx in [0, 1]:
            basename = '{:s}-{:d}'.format(sample_name, cam_idx)
            color_img_path = osp.join(
                dataset_dir, 'color-input', basename + '.png')
            depth_img_path = osp.join(
                dataset_dir, 'depth-input', basename + '.png')
            bg_color_img_path = osp.join(
                dataset_dir, 'color-background', basename + '.png')
            bg_depth_img_path = osp.join(
                dataset_dir, 'depth-background', basename + '.png')
            cam_intrinsics_path = osp.join(
                dataset_dir, 'camera-intrinsics', basename + '.txt')
            cam_pose_path = osp.join(
                dataset_dir, 'camera-pose', basename + '.txt')

            suction_img_path = None
            if grasp_type == 'suction':
                suction_img_path = osp.join(
                    dataset_dir, 'label', basename + '.png')

            if not osp.exists(color_img_path):
                continue

            # Read RGB-D image files
            color_img = skimage.io.imread(color_img_path) / 255
            depth_img = skimage.io.imread(depth_img_path) / 10000
            bg_color_img = skimage.io.imread(bg_color_img_path) / 255
            bg_depth_img = skimage.io.imread(bg_depth_img_path) / 10000
            cam_intrinsics = np.loadtxt(cam_intrinsics_path)
            cam_pose = np.loadtxt(cam_pose_path)

            suction_img = None
            if suction_img_path is not None:
                suction_img = skimage.io.imread(suction_img_path) / 255

            heightmap_color, heightmap, missing_heightmap, heightmap_suction \
                = grasp_fusion.utils.get_heightmap(
                    color_img,
                    depth_img,
                    bg_color_img,
                    bg_depth_img,
                    cam_intrinsics,
                    cam_pose,
                    grid_origin,
                    bin_rot,
                    suction_img,
                    voxel_size=0.002,
                )

            heightmaps_color.append(heightmap_color)
            heightmaps.append(heightmap)
            missing_heightmaps.append(missing_heightmap)
            heightmaps_suction.append(heightmap_suction)
            del heightmap_color, heightmap, missing_heightmap, \
                heightmap_suction

        n_view = len(heightmaps_color)
        assert n_view == len(heightmaps) == len(missing_heightmaps) \
                      == len(heightmaps_suction)
        assert n_view in [1, 2]

        if n_view == 2:
            heightmap_color = np.maximum(
                heightmaps_color[0], heightmaps_color[1]
            )
            heightmap = np.maximum(heightmaps[0], heightmaps[1])
            missing_heightmap = missing_heightmaps[0] & missing_heightmaps[1]
            if grasp_type == 'suction':
                heightmap_suction = np.maximum(
                    heightmaps_suction[0], heightmaps_suction[1]
                )
        else:
            heightmap_color = heightmaps_color[0]
            heightmap = heightmaps[0]
            missing_heightmap = missing_heightmaps[0]
            if grasp_type == 'suction':
                heightmap_suction = heightmaps_suction[0]
        del heightmaps_color, heightmaps, missing_heightmaps

        color_data, depth_data \
            = grasp_fusion.utils.heightmap_postprocess(
                heightmap_color,
                heightmap,
                missing_heightmap,
            )

        if grasp_type == 'suction':
            heightmap_suction = heightmap_suction.reshape(
                heightmap.shape[0], heightmap.shape[1],
            )
            suction_data = np.zeros((224, 320), dtype=np.uint8)
            suction_data[12:212, 10:310] = heightmap_suction * 255

        heightmap_color_file = osp.join(
            heightmap_color_dir, sample_name + '.png')
        heightmap_depth_file = osp.join(
            heightmap_depth_dir, sample_name + '.png')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(heightmap_color_file, color_data)
            skimage.io.imsave(heightmap_depth_file, depth_data)
            if grasp_type == 'suction':
                heightmap_suction_file = osp.join(
                    heightmap_suction_dir, sample_name + '.png')
                skimage.io.imsave(heightmap_suction_file, suction_data)


if __name__ == '__main__':
    for grasp_type in ['pinch', 'suction']:
        get_heightmaps(grasp_type)
