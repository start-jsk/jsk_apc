import os.path as osp

import instance_occlsegm_lib


def test_meshplot():
    here = osp.dirname(osp.realpath(__file__))

    filename = osp.join(here, 'data/cube.off')
    verts, faces = instance_occlsegm_lib.io.load_off(filename)

    instance_occlsegm_lib.io.meshplot(verts, faces)
