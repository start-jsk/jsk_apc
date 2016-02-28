#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import json

import jsk_apc2015_common


parser = argparse.ArgumentParser()
parser.add_argument('json_id')
args = parser.parse_args()
json_id = args.json_id

N = 2
target_bin = 'h'
objects = jsk_apc2015_common.get_object_list()
json_data = {}
candidates = []
for i in xrange(N):
    i_obj = random.randint(0, len(objects) - 1)
    candidates.append(objects[i_obj])
i_target = random.randint(0, len(candidates) - 1)
target_obj = candidates[i_target]

json_data = {
    'bin_contents': {
        'bin_{0}'.format(target_bin.upper()): candidates,
    },
    'work_order': [
        {
            'bin': 'bin_{0}'.format(target_bin.upper()),
            'item': target_obj,
        },
    ],
}

json.dump(json_data, open('{0}.json'.format(json_id), 'wb'), indent=2)