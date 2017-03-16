#!/usr/bin/env python

import os.path as osp

import rospkg


PKG_PATH = rospkg.RosPack().get_path('jsk_arc2017_common')

object_names = ['__background__']
with open(osp.join(PKG_PATH, 'data/names/objects.txt')) as f:
    object_names += [x.strip() for x in f]
object_names.append('__shelf__')

for obj_id, obj in enumerate(object_names):
    print('%2d: %s' % (obj_id, obj))
