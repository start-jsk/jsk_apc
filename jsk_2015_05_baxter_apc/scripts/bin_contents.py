#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json

import rospy
from jsk_2014_picking_challenge.msg import BinContents, BinContentsArray


def main():
    json_file = rospy.get_param('~json')
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    with open(json_file, 'r') as f:
        bin_contents = json.load(f)['bin_contents']

    msg = BinContentsArray()
    for bin_, objects in bin_contents.items():
        bin_ = bin_.split('_')[1].lower()  # bin_A -> a
        msg.array.append(BinContents(bin=bin_, objects=objects))

    pub = rospy.Publisher('/bin_contents', BinContentsArray)
    rate = rospy.Rate(rospy.get_param('rate', 1))
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('bin_contents')
    main()
