#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import cv2
import numpy as np

import rospy
import cv_bridge
import dynamic_reconfigure.server
from sensor_msgs.msg import Image
from jsk_2014_picking_challenge.cfg import DilateImageConfig


class DilateImage(object):
    def __init__(self):
        dynamic_reconfigure.server.Server(DilateImageConfig,
                                          self._cb_dyn_reconfig)
        rospy.Subscriber('~input', Image, self._cb_img)
        self._pub = rospy.Publisher('~output', Image, queue_size=1)
        self.img = None
        self.stamp = None

    def _cb_dyn_reconfig(self, config, level):
        self.winsize_col = config['winsize_col']
        self.winsize_row = config['winsize_row']
        self.iterations = config['iterations']
        return config

    def _cb_img(self, msg):
        bridge = cv_bridge.CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.stamp= msg.header.stamp

    def spin_once(self):
        if self.img is None or self.stamp is None:
            return
        img = self.img
        stamp = self.stamp
        out = cv2.dilate(img, np.ones((self.winsize_row, self.winsize_col)),
                         iterations=self.iterations)
        bridge = cv_bridge.CvBridge()
        imgmsg = bridge.cv2_to_imgmsg(out, encoding='mono8')
        imgmsg.header.stamp = stamp
        self._pub.publish(imgmsg)

    def spin(self):
        rate = rospy.Rate(rospy.get_param('rate', 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('dilate_image')
    dilate = DilateImage()
    dilate.spin()

