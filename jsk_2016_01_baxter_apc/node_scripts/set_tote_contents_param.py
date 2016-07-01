#!/usr/bin/env python

import argparse
import json
import sys

import rospy


if __name__ == '__main__':
    rospy.init_node('set_tote_contents_param')

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args(args=rospy.myargv()[1:])

    json_file = args.json_file

    data = json.load(open(json_file))
    rospy.set_param('~tote_contents', data['tote_contents'])
    sys.exit(0)
