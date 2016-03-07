#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

import yaml
import rospkg


rp = rospkg.RosPack()
PKG = 'jsk_apc2016_common'


def get_object_data():
    fname = osp.join(rp.get_path(PKG), 'data/object_data.yaml')
    data = yaml.load(open(fname))
    return data
