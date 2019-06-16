import os.path as osp

import numpy as np

import grasp_fusion_lib


here = osp.dirname(osp.realpath(__file__))


def test_load_pcd():
    filename = osp.join(here, 'data/bunny.pcd')
    points = grasp_fusion_lib.io.load_pcd(filename)

    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3  # xyz
