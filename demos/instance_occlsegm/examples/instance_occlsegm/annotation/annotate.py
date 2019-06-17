#!/usr/bin/env python

import os
import os.path as osp
import subprocess


here = osp.dirname(osp.abspath(__file__))
cmd = 'labelme --config {config} {data}'.format(
    config=osp.join(here, 'labelmerc'),
    data=osp.relpath(
        osp.join(here, 'data/annotation_raw_data/20180730'), os.curdir
    ),
)
print('+ {cmd}'.format(cmd=cmd))
subprocess.call(cmd, shell=True)
