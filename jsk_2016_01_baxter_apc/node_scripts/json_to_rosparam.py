#!/usr/bin/env python

import argparse
import json
import sys

import rospy


if __name__ == '__main__':
    rospy.init_node('json_to_rosparam')

    json_file = rospy.get_param('~json')
    key = rospy.get_param('~key')

    data = json.load(open(json_file))
    rospy.set_param('~param', data[key])
    sys.exit(0)
