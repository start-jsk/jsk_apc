import skimage.data

import grasp_fusion_lib


def test_imgplot():
    img = skimage.data.coffee()
    grasp_fusion_lib.io.imgplot(img)
