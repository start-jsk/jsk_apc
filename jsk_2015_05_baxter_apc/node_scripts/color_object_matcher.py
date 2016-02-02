#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
this script will test classification by the color_histogram
"""

import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import ColorHistogram
from jsk_2015_05_baxter_apc.msg import ObjectRecognition
import jsk_apc2015_common
from common import ObjectMatcher
from color_histogram_features import ColorHistogramFeatures
import cv_bridge

import numpy as np


class ColorObjectMatcher(ObjectMatcher):

    def __init__(self):
        super(ColorObjectMatcher, self).__init__('/color_object_matcher')
        self._pub_recog = rospy.Publisher('~output', ObjectRecognition,
                                          queue_size=1)
        self._pub_debug = rospy.Publisher(
            '~debug', ColorHistogram, queue_size=1)
        self.query_image = None
        self.estimator = ColorHistogramFeatures()
        self.estimator.load_data()
        rospy.Subscriber('~input', Image, self._predict)

    def _predict(self, msg):
        # convert image
        bridge = cv_bridge.CvBridge()
        input_image = bridge.imgmsg_to_cv2(msg, 'rgb8')
        # compute histogram
        hist = self.estimator.color_hist(input_image)
        self._pub_debug.publish(
            ColorHistogram(header=query_image.header, histogram=hist))
        # predict
        proba = self.estimator.predict(input_image)[0]
        objects = jsk_apc2015_common.get_object_list()
        matched_idx = np.argmax(proba)
        # prepare message
        res = ObjectRecognition()
        res.header = msg.header
        res.matched = objects[matched_idx]
        res.probability = proba[matched_idx]
        res.candidates = objects
        res.probabilities = proba
        return res


def main():
    rospy.init_node('color_object_matcher')
    matcher = ColorObjectMatcher()
    rospy.spin()


if __name__ == "__main__":
    main()
