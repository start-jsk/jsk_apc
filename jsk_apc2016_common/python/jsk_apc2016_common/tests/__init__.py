#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal
from nose.tools import assert_true
import os.path as osp

import numpy as np

import jsk_apc2016_common


_this_dir = osp.dirname(osp.realpath(osp.abspath(__file__)))


def test_get_object_data():
    obj_data = jsk_apc2016_common.get_object_data()
    obj_names = map(lambda d: d['name'], obj_data)
    assert_true(isinstance(obj_data, list))
    assert_equal(39, len(obj_data))
    assert_equal(sorted(obj_names), obj_names)
    for d in obj_data:
        assert_true(isinstance(d, dict))
        assert_true('name' in d)
        assert_true('weight' in d)
        assert_true('graspability' in d)
        assert_true('stock' in d)


def test_get_object_data_2015():
    obj_data = jsk_apc2016_common.get_object_data_2015()
    assert_true(isinstance(obj_data, list))
    for d in obj_data:
        assert_true(isinstance(d, dict))
        assert_true('name' in d)


def test__load_stow_json():
    json_file = osp.join(_this_dir, 'data', 'stow_layout_1.json')
    bin_contents, work_order = jsk_apc2016_common._load_stow_json(json_file)
    assert_true(isinstance(bin_contents, dict))
    for bin_, contents in bin_contents.items():
        assert_true(isinstance(bin_, basestring))
        assert_true(isinstance(contents, list))
    assert_true(isinstance(work_order, list))


def test_visualize_stow_contents():
    json_file = osp.join(_this_dir, 'data', 'stow_layout_1.json')
    _, work_order = jsk_apc2016_common._load_stow_json(json_file)
    img = jsk_apc2016_common.visualize_stow_contents(work_order)
    assert_equal(img.shape, (960, 1280, 3))
    assert_equal(img.dtype, np.uint8)


def test_visualize_stow_json():
    json_file = osp.join(_this_dir, 'data', 'stow_layout_1.json')
    img = jsk_apc2016_common.visualize_stow_json(json_file)
    assert_equal(img.shape, (3985, 2067, 3))
    assert_equal(img.dtype, np.uint8)
