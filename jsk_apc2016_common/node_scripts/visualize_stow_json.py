#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import cv_bridge
import rospy
from sensor_msgs.msg import Image

import jsk_apc2016_common


def publish_cb(event):
    imgmsg.header.stamp = rospy.Time.now()
    pub.publish(imgmsg)


if __name__ == '__main__':
    rospy.init_node('visualize_stow_json')
    pub = rospy.Publisher('~output', Image, queue_size=10)

    parser = argparse.ArgumentParser()
    parser.add_argument('json',
                        help='JSON file with bin_contents and work_order')
    args = parser.parse_args(rospy.myargv()[1:])
    json = args.json

    img = jsk_apc2016_common.visualize_stow_json(json)

    br = cv_bridge.CvBridge()
    imgmsg = br.cv2_to_imgmsg(img, encoding='bgr8')

    timer = rospy.Timer(rospy.Duration(0.1), publish_cb)
    rospy.spin()
