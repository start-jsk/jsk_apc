#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal

import jsk_apc2015_common.data


def test_object_list():
    objects = jsk_apc2015_common.data.object_list()
    assert_equal(25, len(objects))
