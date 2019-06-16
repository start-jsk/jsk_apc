import cv2
import skimage.data

import grasp_fusion_lib


def test_resize():
    for interpolation in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
        _test_resize(interpolation)


def _test_resize(interpolation):
    img = skimage.data.astronaut()
    H_dst, W_dst = 480, 640

    ret = grasp_fusion_lib.image.resize(img, height=H_dst, width=W_dst,
                                        interpolation=interpolation)
    assert ret.dtype == img.dtype
    assert ret.shape == (H_dst, W_dst, 3)

    ret = grasp_fusion_lib.image.resize(
        img, height=H_dst, interpolation=interpolation)
    hw_ratio = 1. * img.shape[1] / img.shape[0]
    W_expected = int(round(1 / hw_ratio * H_dst))
    assert ret.dtype == img.dtype
    assert ret.shape == (H_dst, W_expected, 3)

    scale = 0.3
    ret = grasp_fusion_lib.image.resize(img, fy=scale, fx=scale,
                                        interpolation=interpolation)
    assert ret.dtype == img.dtype
    H_expected = int(round(img.shape[0] * 0.3))
    W_expected = int(round(img.shape[1] * 0.3))
    assert ret.shape == (H_expected, W_expected, 3)

    scale = 0.3
    ret = grasp_fusion_lib.image.resize(
        img, fy=scale, interpolation=interpolation)
    assert ret.dtype == img.dtype
    assert ret.shape == (H_expected, W_expected, 3)
