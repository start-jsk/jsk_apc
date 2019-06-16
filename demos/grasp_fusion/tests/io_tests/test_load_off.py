import os.path as osp

import numpy as np

import grasp_fusion_lib


here = osp.dirname(osp.realpath(__file__))


def test_load_off():
    filename = osp.join(here, 'data/cube.off')
    verts, faces = grasp_fusion_lib.io.load_off(filename)

    assert isinstance(verts, np.ndarray)
    assert verts.ndim == 2
    assert verts.shape[1] == 3
    assert verts.dtype == np.float64
    assert isinstance(faces, np.ndarray)
    assert faces.ndim == 2
    assert faces.shape[1] == 4
    assert faces.dtype == np.int64
