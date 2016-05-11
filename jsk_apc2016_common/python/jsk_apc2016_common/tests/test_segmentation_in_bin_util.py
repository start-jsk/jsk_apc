from nose.tools import assert_equal

import jsk_apc2016_common.segmentation_in_bin_util as sib_util


def test_segmentation_in_bin_util():
    points = [1, 2, 3]
    bbox_dimensions = [10, 10, 10]
    feature = sib_util.get_spatial_feature(points, bbox_dimensions)
    assert_equal((2.0, 8.0), feature)
