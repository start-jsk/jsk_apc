import skimage.data

import instance_occlsegm_lib


def test_imgplot():
    img = skimage.data.coffee()
    instance_occlsegm_lib.io.imgplot(img)
