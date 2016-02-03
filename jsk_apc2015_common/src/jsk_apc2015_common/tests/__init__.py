#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal

import jsk_apc2015_common


def test_get_object_list():
    objects = jsk_apc2015_common.get_object_list()
    assert_equal(25, len(objects))
