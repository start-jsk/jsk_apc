#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Color cutback for RGB decomposed image."""

import rospy
import cv_bridge
import dynamic_reconfigure.server
from jsk_2014_picking_challenge.cfg import ColorCutbackConfig
from sensor_msgs.msg import Image


class ColorCutback(object):
    def __init__(self):
        self.threshold = rospy.get_param('~threshold', 30)
        self.img = {}
        self.stamp = None
        dynamic_reconfigure.server.Server(ColorCutbackConfig,
                                          self._cb_dynamic_reconfigure)
        self._init_publishers()
        self._init_subscribers()
        rospy.wait_for_message('~input/reference', Image)

    def _cb_dynamic_reconfigure(self, config, level):
        """Callback function of dynamic reconfigure server"""
        self.threshold = config['threshold']
        return config

    def _init_publishers(self):
        pub_tmpl = lambda c: rospy.Publisher('~output/{}'.format(c), Image,
                                             queue_size=1)
        self._pub = {c: pub_tmpl(c) for c in
                     ['red', 'green', 'blue', 'hue', 'saturation', 'value']}

    def _init_subscribers(self):
        rospy.Subscriber('~input/red', Image,
                         lambda msg: self._cb_input(msg, 'red'))
        rospy.Subscriber('~input/green', Image,
                         lambda msg: self._cb_input(msg, 'green'))
        rospy.Subscriber('~input/blue', Image,
                         lambda msg: self._cb_input(msg, 'blue'))
        rospy.Subscriber('~input/hue', Image,
                         lambda msg: self._cb_input(msg, 'hue'))
        rospy.Subscriber('~input/saturation', Image,
                         lambda msg: self._cb_input(msg, 'saturation'))
        rospy.Subscriber('~input/value', Image,
                         lambda msg: self._cb_input(msg, 'value'))
        rospy.Subscriber('~input/reference', Image,
                         lambda msg: self._cb_input(msg, 'reference'))

    def _cb_input(self, msg, attr):
        bridge = cv_bridge.CvBridge()
        self.stamp = msg.header.stamp
        self.img[attr] = bridge.imgmsg_to_cv2(msg)

    def _color_spin_once(self, color):
        if color not in self.img:
            return
        img = self.img[color]
        img_ref = self.img['reference']
        threshold = self.threshold
        # color cutback with threshold
        # converted image size: height=1, width=*
        img_cutback = img[img_ref > threshold].reshape((1, -1))
        bridge = cv_bridge.CvBridge()
        imgmsg = bridge.cv2_to_imgmsg(img_cutback, encoding='mono8')
        imgmsg.header.stamp = self.stamp
        self._pub[color].publish(imgmsg)

    def spin_once(self):
        for c in ['red', 'green', 'blue', 'hue', 'saturation', 'value']:
            self._color_spin_once(color=c)

    def spin(self):
        rate = rospy.Rate(rospy.get_param("rate", 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('color_cutback')
    cutback = ColorCutback()
    cutback.spin()

