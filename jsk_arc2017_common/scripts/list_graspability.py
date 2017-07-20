#!/usr/bin/env python

import jsk_arc2017_common

graspability = jsk_arc2017_common.get_object_graspability()
for obj_id, obj in enumerate(graspability):
    print('{:02}: {}'.format(obj_id+1, obj))
    for style in graspability[obj]:
        print('    {}: {}'.format(style, graspability[obj][style]))
