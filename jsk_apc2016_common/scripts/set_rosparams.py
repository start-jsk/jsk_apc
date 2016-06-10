#!/usr/bin/env python

"""A script that changes rosparams of visualize_json and publish_bin_info
"""

import sh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json')
args = parser.parse_args()

rosparam_cmd = sh.Command('rosparam')

rosparam_cmd('set', '/visualize_json/json', args.json)
rosparam_cmd('set', '/set_bin_param/json', args.json)
