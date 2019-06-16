import os.path as osp

import numpy as np

import grasp_fusion_lib


this_dir = osp.dirname(osp.realpath(__file__))


def test_colorize_depth():
    depth_file = osp.join(this_dir, 'data/depth.npz')
    depth = np.load(depth_file)['arr_0']

    min_value = np.nanmin(depth)
    max_value = np.nanmax(depth)
    depth_viz = grasp_fusion_lib.image.colorize_depth(
        depth, min_value, max_value)

    assert depth_viz.shape[:2] == depth.shape
