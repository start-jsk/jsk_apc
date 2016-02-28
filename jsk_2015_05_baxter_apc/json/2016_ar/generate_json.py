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
abandon_objects = [
    'genuine_joe_plastic_stir_sticks',
    'cheezit_big_original',
    'rolodex_jumbo_pencil_cup',
    'champion_copper_plus_spark_plug',
    'oreo_mega_stuf',
]
objects = jsk_apc2015_common.get_object_list()

target_obj = None
while (target_obj is None) or (target_obj in abandon_objects):
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