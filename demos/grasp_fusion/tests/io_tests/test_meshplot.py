import os.path as osp

import grasp_fusion_lib


def test_meshplot():
    here = osp.dirname(osp.realpath(__file__))

    filename = osp.join(here, 'data/cube.off')
    verts, faces = grasp_fusion_lib.io.load_off(filename)

    grasp_fusion_lib.io.meshplot(verts, faces)
