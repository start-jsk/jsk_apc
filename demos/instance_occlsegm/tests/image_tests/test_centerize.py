import skimage.data

import instance_occlsegm_lib


def test_centerize():
    img = skimage.data.astronaut()
    dst_shape = (480, 640)
    img_c, mask_c = instance_occlsegm_lib.image.centerize(
        img, dst_shape, return_mask=True)
    assert img_c.shape == (dst_shape[0], dst_shape[1], 3)
    assert img_c.dtype == img.dtype
    assert mask_c.shape == dst_shape
    assert mask_c.dtype == bool
