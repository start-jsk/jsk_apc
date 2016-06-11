#!/usr/bin/env python

"""A script that changes all json rosparams related to data collecting routine
"""

import sh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json')
args = parser.parse_args()

rosparam_cmd = sh.Command('rosparam')

rosparam_cmd('set', '/visualize_json/json', args.json)
rosparam_cmd('set', '/set_bin_param/json', args.json)
