#!/usr/bin/env python

"""A script that changes rosparams of visualize_json and publish_bin_info
"""

import sh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json')
args = parser.parse_args()

vis_json_cmd = sh.Command('rosparam', 'set', '/visualize_json/json', args.json)
set_bin_param_cmd = sh.Command(
    'rosparam', 'set', '/set_bin_param/json', args.json)

vis_json_cmd()
set_bin_param_cmd()
