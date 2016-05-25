#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

import cv2
from nose.tools import assert_equal
from nose.tools import assert_true
import numpy as np

import jsk_apc2015_common
import rospkg


_this_dir = osp.dirname(osp.realpath(osp.abspath(__file__)))


def test_get_object_list():
    objects = jsk_apc2015_common.get_object_list()
    assert_equal(25, len(objects))


def test_load_json():
    json_file = osp.join(_this_dir, 'data', 'f2.json')
    bin_contents, work_order = jsk_apc2015_common.load_json(json_file)
    assert_true(isinstance(bin_contents, dict))
    assert_equal(bin_contents.keys(), ['f'])
    assert_true(isinstance(work_order, dict))
    assert_equal(work_order.keys(), ['f'])


def test__get_tile_shape():
    assert_equal((0, 0), jsk_apc2015_common._get_tile_shape(img_num=0))
    assert_equal((1, 1), jsk_apc2015_common._get_tile_shape(img_num=1))
    assert_equal((2, 1), jsk_apc2015_common._get_tile_shape(img_num=2))
    assert_equal((2, 2), jsk_apc2015_common._get_tile_shape(img_num=3))
    assert_equal((2, 2), jsk_apc2015_common._get_tile_shape(img_num=4))
    assert_equal((3, 2), jsk_apc2015_common._get_tile_shape(img_num=5))
    assert_equal((3, 2), jsk_apc2015_common._get_tile_shape(img_num=6))
    assert_equal((3, 3), jsk_apc2015_common._get_tile_shape(img_num=7))


def test_visualize_bin_contents():
    json_file = osp.join(_this_dir, 'data', 'f2.json')
    bin_contents, work_order = jsk_apc2015_common.load_json(json_file)
    # work_order is None
    img = jsk_apc2015_common.visualize_bin_contents(bin_contents)
    assert_equal(img.shape, (2435, 2067, 3))
    assert_equal(img.dtype, np.uint8)
    # work_order is not None
    img = jsk_apc2015_common.visualize_bin_contents(bin_contents, work_order)
    assert_equal(img.shape, (2435, 2067, 3))
    assert_equal(img.dtype, np.uint8)
    # extra_img_paths are passed
    rp = rospkg.RosPack()
    pkg_path = rp.get_path('jsk_apc2015_common')
    extra_img_paths = {
        'safety_works_safety_glasses':
        osp.join(pkg_path, 'models/safety_works_safety_glasses/image.jpg')
    }
    img = jsk_apc2015_common.visualize_bin_contents(
        bin_contents, extra_img_paths=extra_img_paths)
    assert_equal(img.shape, (2435, 2067, 3))
    assert_equal(img.dtype, np.uint8)
