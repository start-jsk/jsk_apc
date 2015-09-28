#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division
from collections import Counter

import cv2
import numpy as np

import rospy
import cv_bridge
import message_filters
from sensor_msgs.msg import Image
from posedetection_msgs.msg import ImageFeature0D
from jsk_2015_05_baxter_apc.msg import ProbabilisticLabels


class FeatureLabelObjectDetector(object):
    def __init__(self):
        rospy.Subscriber('~input/label', Image, self._cb_label)
        rospy.Subscriber('~input/mask', Image, self._cb_mask)
        rospy.Subscriber('~input/ImageFeature0D', ImageFeature0D,
                         self._cb_features)
        self._pub = rospy.Publisher('~output', ProbabilisticLabels,
                                    queue_size=1)
        self._pub_label = rospy.Publisher('~output/label', Image, queue_size=1)
        self._pub_masks = {}
        self.label_msg = None
        self.mask_msg = None
        self.feature_msg = None

    def _cb_label(self, label_msg):
        self.label_msg = label_msg

    def _cb_mask(self, mask_msg):
        self.mask_msg = mask_msg

    def _cb_features(self, feature0d_msg):
        self.feature_msg = feature0d_msg.features

    def detect(self, feature_pos, label_img, mask):
        feature_pos = feature_pos.astype(int)
        detected_labels = [label_img[y][x] for x, y in feature_pos
                                           if mask[y][x] != 0]
        count = Counter(detected_labels)
        if len(count) == 0:
            return
        labels, proba = zip(*sorted(count.items(), key=lambda x:x[1],
                                    reverse=True))
        proba = np.array(proba) / np.array(proba).sum()
        bridge = cv_bridge.CvBridge()
        self._pub_label.publish(
            bridge.cv2_to_imgmsg(label_img, encoding='32SC1'))
        self._pub.publish(labels=labels, probabilities=proba)
        for label in labels:
            if label not in self._pub_masks:
                self._pub_masks[label] = rospy.Publisher(
                    '~output/label_mask/{0}'.format(label), Image, queue_size=1)
                rospy.sleep(0.1)
            label_mask = label_img.copy().astype(np.uint8)
            label_mask[label_img == label] = 255
            label_mask[label_img != label] = 0
            self._pub_masks[label].publish(
                bridge.cv2_to_imgmsg(label_mask, encoding='mono8'))

    def spin_once(self):
        if ( (self.label_msg is None) or
             (self.mask_msg is None) or
             (self.feature_msg is None) ):
            return
        label_msg = self.label_msg
        mask_msg = self.mask_msg
        feature_msg = self.feature_msg

        positions = np.array(feature_msg.positions).reshape((-1, 2))
        bridge = cv_bridge.CvBridge()
        label_img = bridge.imgmsg_to_cv2(label_msg)
        mask = bridge.imgmsg_to_cv2(mask_msg)
        self.detect(feature_pos=positions, label_img=label_img, mask=mask)

    def spin(self):
        rate = rospy.Rate(rospy.get_param('~rate', 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('feature_label_object_detector')
    detector = FeatureLabelObjectDetector()
    detector.spin()