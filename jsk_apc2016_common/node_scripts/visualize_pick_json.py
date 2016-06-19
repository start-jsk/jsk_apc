#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import cv_bridge
import rospy
from sensor_msgs.msg import Image

import jsk_apc2016_common


def publish_cb(event):
    br = cv_bridge.CvBridge()
    json = rospy.get_param('~json', None)
    if json is None:
        if json_arg is not None:
            json = json_arg
        else:
            rospy.logerr('no json is given')
            return
    img = jsk_apc2016_common.visualize_pick_json(json)
    imgmsg = br.cv2_to_imgmsg(img, encoding='bgr8')
    imgmsg.header.stamp = rospy.Time.now()
    pub.publish(imgmsg)


if __name__ == '__main__':
    rospy.init_node('visualize_pick_json')
    pub = rospy.Publisher('~output', Image, queue_size=10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--json',
                        help='JSON file with bin_contents and work_order',
                        required=False)
    args = parser.parse_args(rospy.myargv()[1:])
    json_arg = args.json
    timer = rospy.Timer(rospy.Duration(1.), publish_cb)
    rospy.spin()
