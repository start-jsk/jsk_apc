#!/usr/bin/env python

from __future__ import print_function

import os.path as osp

import gdown


here = osp.dirname(osp.abspath(__file__))


gdown.cached_download(
    url='https://drive.google.com/uc?id=1Z1GHM1ISabcq-0SbgArTGcgy2vi79bsQ',
    path=osp.join(here, 'annotation_raw_data/20180204.zip'),
    md5='6de52b426fc24594fe24d918b9d1402d',
    postprocess=gdown.extractall,
)

gdown.cached_download(
    url='https://drive.google.com/uc?id=1NB4srQQJVUOjFMCoGg97OFKSsjZylrc-',
    path=osp.join(here, 'annotation_raw_data/20180204_annotated.zip'),
    md5='e2e2c6d49880b4d3610564447ed806b6',
    postprocess=gdown.extractall,
)

raw_data_dirs = [
    '20180204',
    '20180204_annotated',
]

for raw_data_dir in raw_data_dirs:
    raw_data_dir = osp.join(here, 'annotation_raw_data', raw_data_dir)
    print('Checking %s... ' % raw_data_dir, end='')
    assert osp.exists(raw_data_dir), 'Please install %s' % raw_data_dir
    print('Ok')
