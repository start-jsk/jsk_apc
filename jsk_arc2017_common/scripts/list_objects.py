#!/usr/bin/env python

import os.path as osp

import rospkg
import yaml


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')

with open(osp.join(PKG_DIR, 'config/label_names.yaml')) as f:
    object_names = yaml.load(f)

for obj_id, obj in enumerate(object_names):
    print('%2d: %s' % (obj_id, obj))
