#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import cv2

import rospy
import cv_bridge
from sensor_msgs.msg import Image


class ReversedColor(object):
    def __init__(self):
        rospy.Subscriber('~input', Image, self.callback)
        self.pub = rospy.Publisher("~output", Image, queue_size=1)

    def callback(self, msg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(msg)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgmsg = bridge.cv2_to_imgmsg(~img, encoding='mono8')
        imgmsg.header.stamp = msg.header.stamp
        self.pub.publish(imgmsg)


if __name__ == '__main__':
    rospy.init_node("reversed_color")
    rvclr = ReversedColor()
    rospy.spin()
