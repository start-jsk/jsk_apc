from .copy_chainermodel import copy_alex_chainermodel  # NOQA
from .copy_chainermodel import copy_vgg16_chainermodel  # NOQA


def get_APC_pt(arr):
    # calculate APC point
    # drop: -10
    # protrude: -5
    # damage: -5
    pt = 10.0 - 10.0*arr[0] - 5.0*arr[1] - 5.0*arr[2]
    if pt < 0.0:
        pt = 0.0
    return pt
