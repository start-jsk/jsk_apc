#!/usr/bin/env python

import jsk_arc2017_common

for obj_id, obj in enumerate(jsk_arc2017_common.get_label_names()):
    print('{:02}: {}'.format(obj_id, obj))
