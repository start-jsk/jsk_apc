#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from math import pi

import cv2

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from posedetection_msgs.msg import ImageFeature0D


class VisualizeSift(object):
    def __init__(self):
        self.features = None
        self.imgray = None
        rospy.Subscriber('~input/image', Image, self._cb_img)
        rospy.Subscriber('~input/feature', ImageFeature0D, self._cb_features)
        self.pub = rospy.Publisher('~output', Image, queue_size=1)

    def _cb_features(self, msg):
        self.features = msg.features

    def _cb_img(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        self.imgray = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='mono8')

    def spin_once(self):
        if self.features is None:
            return
        features = self.features
        imgray = self.imgray
        bridge = cv_bridge.CvBridge()
        keypoints = []
        for i in xrange(len(features.positions)/2):
            x = features.positions[2*i]
            y = features.positions[2*i+1]
            size = features.scales[i]
            ori = features.orientations[i] / pi * 180
            kp = cv2.KeyPoint(x=x, y=y, _size=size, _angle=ori)
            keypoints.append(kp)
        out = cv2.drawKeypoints(imgray, keypoints,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imgmsg = bridge.cv2_to_imgmsg(out, encoding='bgr8')
        self.pub.publish(imgmsg)

    def spin(self):
        rate = rospy.Rate(rospy.get_param('rate', 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('visualize_sift')
    vis = VisualizeSift()
    vis.spin()

